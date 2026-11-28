import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ==============================================================================
# CONFIGURACIÓN GENERAL
# ==============================================================================
RUTA_IMAGEN = "./VA_P2_Public_Dataset/Images/n0230.jpg"
CARPETA_MASKS = "./VA_P2_Public_Dataset/Masks"
CARPETA_SALIDA = "ejecuciones"
METODO_TAG = "otsu2_hough"

SIGMA_GAUSS = 50
TAM_ROI = 600
TAM_CIERRE = 50


# ==============================================================================
# 1. CARGA DE DATOS Y GUARDADO DE RESULTADOS
# ==============================================================================
def cargarImagenYgt(ruta_imagen, carpeta_masks=CARPETA_MASKS):
    """Carga la imagen RGB y las máscaras binarias de Ground Truth (disco y copa)."""
    img_bgr = cv2.imread(ruta_imagen)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    nombre_img = os.path.splitext(os.path.basename(ruta_imagen))[0]
    ruta_gt = os.path.join(carpeta_masks, f"{nombre_img}.png")
    gt = cv2.imread(ruta_gt, cv2.IMREAD_GRAYSCALE)

    gt_disco = (gt >= 1).astype(np.uint8)
    gt_copa = (gt == 2).astype(np.uint8)
    return nombre_img, img_rgb, gt_disco, gt_copa


def guardarEjecucion(figuras, params=None, carpeta_base=CARPETA_SALIDA):
    """Guarda un diccionario {nombre_archivo: fig} y params.txt en ejecuciones/<nombre>/."""
    partes = [str(params[k]) for k in ("imagen", "metodo_tag") if params and k in params]
    carpeta = os.path.join(carpeta_base, "_".join(partes) if partes else "ejecucion")
    os.makedirs(carpeta, exist_ok=True)

    for nombre_archivo, fig in figuras.items():
        if fig is not None:
            fig.savefig(os.path.join(carpeta, nombre_archivo), dpi=150, bbox_inches="tight")

    if params:
        with open(os.path.join(carpeta, "params.txt"), "w") as f:
            for clave, valor in params.items():
                f.write(f"{clave}: {valor}\n")

    print(f"Ejecución guardada en: {carpeta}")


# ==============================================================================
# 2. PASO 1: LOCALIZACIÓN DEL DISCO Y RECORTE DE LA ROI
# ==============================================================================
"""Localiza el disco mediante máximo tras Gaussiano en (G+R)/2 y recorta la ROI cuadrada."""
def localizarYrecortarRoi(img_rgb, gt_disco, gt_copa, sigma_gauss=SIGMA_GAUSS, tam_roi=TAM_ROI):
    r, g, b = cv2.split(img_rgb)
    gb = (g.astype(np.float32) + b.astype(np.float32)) / 2.0
    gr = (g.astype(np.float32) + r.astype(np.float32)) / 2.0

    suavizado = cv2.GaussianBlur(gr, (0, 0), sigmaX=sigma_gauss)
    _, _, _, (cx, cy) = cv2.minMaxLoc(suavizado)

    h, w = img_rgb.shape[:2]
    mitad = tam_roi // 2
    x1 = max(0, min(cx - mitad, w - tam_roi))
    y1 = max(0, min(cy - mitad, h - tam_roi))
    x2 = min(w, x1 + tam_roi)
    y2 = min(h, y1 + tam_roi)

    return {
        "r": r, "g": g, "b": b, "gb": gb, "gr": gr,
        "centro": (cx, cy),
        "bbox": (x1, y1, x2, y2),
        "roi_rgb": img_rgb[y1:y2, x1:x2],
        "roi_r": r[y1:y2, x1:x2],
        "roi_g": g[y1:y2, x1:x2],
        "roi_gt_disco": gt_disco[y1:y2, x1:x2],
        "roi_gt_copa": gt_copa[y1:y2, x1:x2],
    }


# ==============================================================================
# 3. PASO 2: SEGMENTACIÓN OTSU EN 2 ETAPAS Y CIERRE MORFOLÓGICO
# ==============================================================================
"""Calcula el umbral de Otsu restringido a los píxeles donde mascara_valida es True."""
def calcularUmbralOtsu(imagen, mascara_valida):
    pixeles = imagen[mascara_valida]
    datos = pixeles.reshape(-1, 1) if pixeles.size > 0 else imagen
    umbral, _ = cv2.threshold(datos, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return float(umbral)


def segmentarOtsuDosEtapas(roi_r, roi_g, tam_cierre=TAM_CIERRE):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam_cierre, tam_cierre))

    # Etapa 1: Otsu en canal R (ignorando borde negro R <= 10) + cierre -> Disco.
    umbral_r = calcularUmbralOtsu(roi_r, roi_r > 10)
    mascara_r = (roi_r > umbral_r).astype(np.uint8)
    cierre_r = cv2.morphologyEx(mascara_r, cv2.MORPH_CLOSE, kernel)

    # Etapa 2: Otsu en canal G restringido al disco + cierre -> Copa.
    roi_g_disco = roi_g * cierre_r
    umbral_g = calcularUmbralOtsu(roi_g, cierre_r == 1)
    mascara_g = ((roi_g > umbral_g) & (cierre_r == 1)).astype(np.uint8)
    cierre_g = cv2.morphologyEx(mascara_g, cv2.MORPH_CLOSE, kernel)

    return {
        "umbral_r": umbral_r, "mascara_r": mascara_r, "cierre_r": cierre_r,
        "roi_g_disco": roi_g_disco,
        "umbral_g": umbral_g, "mascara_g": mascara_g, "cierre_g": cierre_g,
    }


# ==============================================================================
# 4. PASO 3: BORDES BINARIOS Y TRANSFORMADA DE HOUGH (CÍRCULOS Y ELIPSES)
# ==============================================================================
"""Extrae el contorno interior (1 píxel) mediante resta morfológica A - (A ⊖ B)."""
def extraerBordesBinarios(mascara_bin):
    kernel_3 = np.ones((3, 3), np.uint8)
    return (mascara_bin - cv2.erode(mascara_bin, kernel_3)).astype(np.uint8)


def houghElipse(bordes_bin, escala=0.15, min_ratio=0.6, paso_b=1.5):
    """
    Transformada de Hough para elipses (Xie & Ji, 2002) sobre bordes binarios.
    Devuelve (xc, yc, a, b, angulo_grados, votos) en coordenadas originales.
    """
    ys, xs = np.nonzero(bordes_bin)
    if xs.size < 10:
        return None

    # Reducir resolución y muestrear puntos de borde para mantener coste acotado
    pts = np.unique(np.column_stack((np.round(xs * escala), np.round(ys * escala))), axis=0)
    if len(pts) > 120:
        pts = pts[np.linspace(0, len(pts) - 1, 120, dtype=int)]

    span = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())
    min_2a, max_2a = max(6.0, span * 0.55), span * 1.15

    px, py = pts[:, 0], pts[:, 1]
    n = len(pts)
    mejor_votos = -1
    mejor_elipse = None

    for i in range(n - 1):
        dist = np.hypot(px[i + 1:] - px[i], py[i + 1:] - py[i])

        for off in np.where((dist >= min_2a) & (dist <= max_2a))[0]:
            j = i + 1 + off
            a = 0.5 * dist[off]
            x0, y0 = 0.5 * (px[i] + px[j]), 0.5 * (py[i] + py[j])
            alpha = np.arctan2(py[j] - py[i], px[j] - px[i])
            ca, sa = np.cos(alpha), np.sin(alpha)

            # Proyección de todos los puntos de borde sobre los ejes candidatos
            rx, ry = px - x0, py - y0
            u = rx * ca + ry * sa
            v = -rx * sa + ry * ca

            u2_a2 = (u * u) / (a * a)
            validos = u2_a2 < 0.99
            if not np.any(validos):
                continue

            # Acumulador 1D para el semieje menor b
            b_vals = np.sqrt((v[validos] ** 2) / (1.0 - u2_a2[validos]))
            b_valid = b_vals[(b_vals >= min_ratio * a) & (b_vals <= a * 1.05)]
            if b_valid.size == 0:
                continue

            counts = np.bincount(np.floor(b_valid / paso_b).astype(np.int32))
            idx_max = int(np.argmax(counts))
            votos = int(counts[idx_max])

            if votos > mejor_votos:
                mejor_votos = votos
                b_est = (idx_max + 0.5) * paso_b
                mejor_elipse = (
                    x0 / escala, y0 / escala,
                    a / escala, b_est / escala,
                    float(np.degrees(alpha)), votos,
                )

    return mejor_elipse


def houghCirculo(bordes_bin, escala=0.2):
    """
    Transformada de Hough estándar para círculos (xc, yc, r) sobre bordes binarios.
    Devuelve (xc, yc, r, votos) en coordenadas originales.
    """
    h, w = bordes_bin.shape
    nh, nw = max(1, int(round(h * escala))), max(1, int(round(w * escala)))
    ys, xs = np.nonzero(bordes_bin)
    if xs.size < 10:
        return None

    pts = np.unique(
        np.column_stack((np.round(xs * escala), np.round(ys * escala))).astype(np.int32),
        axis=0,
    )
    px, py = pts[:, 0], pts[:, 1]
    span = max(px.max() - px.min(), py.max() - py.min())
    r_min = max(4, int(round(span * 0.25)))
    r_max = max(r_min + 1, int(round(span * 0.60)))

    thetas = np.linspace(0, 2 * np.pi, 72, endpoint=False)
    cos_t, sin_t = np.cos(thetas), np.sin(thetas)

    mejor_votos = -1
    mejor_circulo = None

    for r in range(r_min, r_max + 1):
        offs = np.unique(
            np.column_stack((np.round(r * cos_t), np.round(r * sin_t))).astype(np.int32),
            axis=0,
        )
        cx_c = (px[:, None] - offs[None, :, 0]).ravel()
        cy_c = (py[:, None] - offs[None, :, 1]).ravel()
        valid = (cx_c >= 0) & (cx_c < nw) & (cy_c >= 0) & (cy_c < nh)

        acc = np.zeros((nh, nw), dtype=np.int32)
        np.add.at(acc, (cy_c[valid], cx_c[valid]), 1)

        cy_m, cx_m = divmod(int(np.argmax(acc)), nw)
        votos = int(acc[cy_m, cx_m])

        if votos > mejor_votos:
            mejor_votos = votos
            mejor_circulo = (cx_m / escala, cy_m / escala, r / escala, votos)

    return mejor_circulo


def dibujarElipse(imagen, elipse, color, grosor=-1):
    if elipse is None:
        return
    xc, yc, a, b, angulo, _ = elipse
    cv2.ellipse(imagen, (int(round(xc)), int(round(yc))), (max(1, int(round(a))), max(1, int(round(b)))), angulo, 0, 360, color, grosor)


def dibujarCirculo(imagen, circulo, color, grosor=-1):
    if circulo is None:
        return
    xc, yc, r, _ = circulo
    cv2.circle(imagen, (int(round(xc)), int(round(yc))), max(1, int(round(r))), color, grosor)


def ajustarHough(cierre_r, cierre_g):
    """Extrae bordes binarios y ajusta círculos y elipses de Hough para disco y copa."""
    bordes_r = extraerBordesBinarios(cierre_r)
    bordes_g = extraerBordesBinarios(cierre_g)

    circ_disco = houghCirculo(bordes_r)
    circ_copa = houghCirculo(bordes_g)
    elip_disco = houghElipse(bordes_r)
    elip_copa = houghElipse(bordes_g)

    mascara_circ_r = np.zeros_like(cierre_r, dtype=np.uint8)
    mascara_circ_g = np.zeros_like(cierre_g, dtype=np.uint8)
    dibujarCirculo(mascara_circ_r, circ_disco, 1, grosor=-1)
    dibujarCirculo(mascara_circ_g, circ_copa, 1, grosor=-1)

    mascara_elip_r = np.zeros_like(cierre_r, dtype=np.uint8)
    mascara_elip_g = np.zeros_like(cierre_g, dtype=np.uint8)
    dibujarElipse(mascara_elip_r, elip_disco, 1, grosor=-1)
    dibujarElipse(mascara_elip_g, elip_copa, 1, grosor=-1)

    return {
        "bordes_r": bordes_r, "bordes_g": bordes_g,
        "circulo_disco": circ_disco, "circulo_copa": circ_copa,
        "elipse_disco": elip_disco, "elipse_copa": elip_copa,
        "mascara_circ_r": mascara_circ_r, "mascara_circ_g": mascara_circ_g,
        "mascara_elip_r": mascara_elip_r, "mascara_elip_g": mascara_elip_g,
    }


# ==============================================================================
# 5. VISUALIZACIÓN (5 VENTANAS DE RESULTADOS PARCIALES)
# ==============================================================================
def finalizarFigura(fig, axes):
    for ax in np.ravel(axes):
        ax.axis("off")
    fig.tight_layout()
    return fig


def combinarMascarasVis(disco, copa):
    """Convierte dos máscaras binarias (disco, copa) a imagen gris (0, 128, 255)."""
    vis = np.zeros_like(disco, dtype=np.uint8)
    vis[disco == 1] = 128
    vis[copa == 1] = 255
    return vis


def anadirLeyenda(ax, etiqueta_disco, etiqueta_copa):
    leyenda = [
        Line2D([0], [0], color=(1, 1, 0), lw=2, label=etiqueta_disco),
        Line2D([0], [0], color=(0, 1, 1), lw=2, label=etiqueta_copa),
        Line2D([0], [0], color=(1, 0, 1), lw=2, label="Ref. Disco"),
        Line2D([0], [0], color=(0, 1, 0), lw=2, label="Ref. Copa"),
    ]
    ax.legend(handles=leyenda, loc="upper right", fontsize=9, framealpha=0.8)


def crearFiguras(img_rgb, loc, otsu, hough, sigma_gauss=SIGMA_GAUSS):
    """Construye las 5 figuras del flujo completo para visualizar y guardar."""
    cx, cy = loc["centro"]
    x1, y1, x2, y2 = loc["bbox"]
    roi_rgb = loc["roi_rgb"]

    contornos_gt_disco, _ = cv2.findContours(loc["roi_gt_disco"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contornos_gt_copa, _ = cv2.findContours(loc["roi_gt_copa"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Ventana 1: Original, canales y localización de ROI ---
    img_roi_loc = img_rgb.copy()
    cv2.rectangle(img_roi_loc, (x1, y1), (x2, y2), (0, 255, 255), 8)
    cv2.circle(img_roi_loc, (cx, cy), 15, (255, 0, 0), -1)

    fig1, axes1 = plt.subplots(2, 3, figsize=(15, 9), num="Original y canales")
    paneles1 = [
        (img_rgb, "Original (RGB)", None),
        (loc["r"], "Canal R", "gray"),
        (loc["g"], "Canal G", "gray"),
        (loc["b"], "Canal B", "gray"),
        (loc["gb"], "Combinación (G + B) / 2", "gray"),
        (img_roi_loc, f"Localización ROI (Gauss σ={sigma_gauss})", None),
    ]
    for ax, (img, titulo, cmap) in zip(axes1.ravel(), paneles1):
        ax.imshow(img, cmap=cmap)
        ax.set_title(titulo)
    finalizarFigura(fig1, axes1)

    # --- Ventana 2: Máscaras Otsu y cierres en la ROI ---
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9), num="Máscaras Otsu y cierres (ROI)")
    paneles2 = [
        (loc["roi_r"], "ROI - Canal R (Disco)", None),
        (otsu["mascara_r"], f"Otsu Canal R (umbral={otsu['umbral_r']:.0f})", (0, 1)),
        (otsu["cierre_r"], "Cierre Canal R (Disco)", (0, 1)),
        (otsu["roi_g_disco"], "ROI - Canal G (dentro del disco)", None),
        (otsu["mascara_g"], f"Otsu Canal G en disco (umbral={otsu['umbral_g']:.0f})", (0, 1)),
        (otsu["cierre_g"], "Cierre Canal G (Copa)", (0, 1)),
    ]
    for ax, (img, titulo, rango) in zip(axes2.ravel(), paneles2):
        kw = {"vmin": rango[0], "vmax": rango[1]} if rango else {}
        ax.imshow(img, cmap="gray", **kw)
        ax.set_title(titulo)
    finalizarFigura(fig2, axes2)

    # --- Ventana 3: ROI, máscaras Otsu con cierre y comparación con GT ---
    contornos_pred_r, _ = cv2.findContours(otsu["cierre_r"], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contornos_pred_g, _ = cv2.findContours(otsu["cierre_g"], cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    comp_otsu = roi_rgb.copy()
    cv2.drawContours(comp_otsu, contornos_gt_disco, -1, (255, 0, 255), 3)
    cv2.drawContours(comp_otsu, contornos_gt_copa, -1, (0, 255, 0), 3)
    cv2.drawContours(comp_otsu, contornos_pred_r, -1, (255, 255, 0), 3)
    cv2.drawContours(comp_otsu, contornos_pred_g, -1, (0, 255, 255), 3)

    fig3, axes3 = plt.subplots(1, 4, figsize=(22, 6), num="ROI y comparación")
    axes3[0].imshow(roi_rgb)
    axes3[0].set_title("ROI - Original")
    axes3[1].imshow(otsu["cierre_r"], cmap="gray", vmin=0, vmax=1)
    axes3[1].set_title("ROI - Disco Otsu (Cierre)")
    axes3[2].imshow(otsu["cierre_g"], cmap="gray", vmin=0, vmax=1)
    axes3[2].set_title("ROI - Copa Otsu (Cierre)")
    axes3[3].imshow(comp_otsu)
    axes3[3].set_title("Comparación de bordes (ROI)")
    anadirLeyenda(axes3[3], "Pred. Disco (Otsu R)", "Pred. Copa (Otsu G)")
    finalizarFigura(fig3, axes3)

    # --- Ventana 4: Bordes binarios y ajuste de elipses con Hough ---
    bordes_r_vis = cv2.cvtColor(hough["bordes_r"] * 255, cv2.COLOR_GRAY2RGB)
    dibujarElipse(bordes_r_vis, hough["elipse_disco"], (255, 255, 0), grosor=3)
    bordes_g_vis = cv2.cvtColor(hough["bordes_g"] * 255, cv2.COLOR_GRAY2RGB)
    dibujarElipse(bordes_g_vis, hough["elipse_copa"], (0, 255, 255), grosor=3)

    fig4, axes4 = plt.subplots(2, 3, figsize=(15, 9), num="Bordes binarios y Hough Elipses (ROI)")
    paneles4 = [
        (hough["bordes_r"], "Bordes binarios Disco (Otsu R)", "gray", (0, 1)),
        (bordes_r_vis, "Hough Elipse sobre bordes Disco", None, None),
        (hough["mascara_elip_r"], "Máscara Elipse Hough (Disco)", "gray", (0, 1)),
        (hough["bordes_g"], "Bordes binarios Copa (Otsu G)", "gray", (0, 1)),
        (bordes_g_vis, "Hough Elipse sobre bordes Copa", None, None),
        (hough["mascara_elip_g"], "Máscara Elipse Hough (Copa)", "gray", (0, 1)),
    ]
    for ax, (img, titulo, cmap, rango) in zip(axes4.ravel(), paneles4):
        kw = {"vmin": rango[0], "vmax": rango[1]} if rango else {}
        ax.imshow(img, cmap=cmap, **kw)
        ax.set_title(titulo)
    finalizarFigura(fig4, axes4)

    # --- Ventana 5: Comparación final Hough Círculos (fila 1) y Elipses (fila 2) vs GT ---
    roi_gt_vis = combinarMascarasVis(loc["roi_gt_disco"], loc["roi_gt_copa"])
    circ_vis = combinarMascarasVis(hough["mascara_circ_r"], hough["mascara_circ_g"])
    elip_vis = combinarMascarasVis(hough["mascara_elip_r"], hough["mascara_elip_g"])

    comp_circ = roi_rgb.copy()
    cv2.drawContours(comp_circ, contornos_gt_disco, -1, (255, 0, 255), 3)
    cv2.drawContours(comp_circ, contornos_gt_copa, -1, (0, 255, 0), 3)
    dibujarCirculo(comp_circ, hough["circulo_disco"], (255, 255, 0), grosor=3)
    dibujarCirculo(comp_circ, hough["circulo_copa"], (0, 255, 255), grosor=3)

    comp_elip = roi_rgb.copy()
    cv2.drawContours(comp_elip, contornos_gt_disco, -1, (255, 0, 255), 3)
    cv2.drawContours(comp_elip, contornos_gt_copa, -1, (0, 255, 0), 3)
    dibujarElipse(comp_elip, hough["elipse_disco"], (255, 255, 0), grosor=3)
    dibujarElipse(comp_elip, hough["elipse_copa"], (0, 255, 255), grosor=3)

    fig5, axes5 = plt.subplots(2, 3, figsize=(18, 11), num="Hough Círculos y Elipses vs Ground Truth (ROI)")
    a5 = axes5.ravel()
    a5[0].imshow(roi_gt_vis, cmap="gray", vmin=0, vmax=255)
    a5[0].set_title("Ground Truth (Disco gris, Copa blanca)")
    a5[1].imshow(circ_vis, cmap="gray", vmin=0, vmax=255)
    a5[1].set_title("Hough Círculos (Disco gris, Copa blanca)")
    a5[2].imshow(comp_circ)
    a5[2].set_title("Comparación de bordes: Hough Círculos vs GT")
    anadirLeyenda(a5[2], "Hough Círculo Disco", "Hough Círculo Copa")

    a5[3].imshow(roi_gt_vis, cmap="gray", vmin=0, vmax=255)
    a5[3].set_title("Ground Truth (Disco gris, Copa blanca)")
    a5[4].imshow(elip_vis, cmap="gray", vmin=0, vmax=255)
    a5[4].set_title("Hough Elipses (Disco gris, Copa blanca)")
    a5[5].imshow(comp_elip)
    a5[5].set_title("Comparación de bordes: Hough Elipses vs GT")
    anadirLeyenda(a5[5], "Hough Elipse Disco", "Hough Elipse Copa")
    finalizarFigura(fig5, axes5)

    return {
        "1_original_y_canales.png": fig1,
        "2_mascaras_y_cierres.png": fig2,
        "3_roi_y_comparacion.png": fig3,
        "4_bordes_y_hough.png": fig4,
        "5_hough_vs_gt.png": fig5,
    }


# ==============================================================================
# 6. FLUJO PRINCIPAL
# ==============================================================================
def imprimirResumen(loc, otsu, hough):
    cx, cy = loc["centro"]
    x1, y1, x2, y2 = loc["bbox"]
    print(f"Centroide (Gauss sigma={SIGMA_GAUSS}): ({cx}, {cy}) | ROI: x=[{x1}, {x2}], y=[{y1}, {y2}]")
    print(f"Otsu Canal R (Disco, R>10)    -> Umbral: {otsu['umbral_r']:.0f}")
    print(f"Otsu Canal G (Copa, en disco) -> Umbral: {otsu['umbral_g']:.0f}")

    for nombre, circ in [("Disco", hough["circulo_disco"]), ("Copa ", hough["circulo_copa"])]:
        if circ is not None:
            print(f"Hough Círculo {nombre} -> centro=({circ[0]:.1f}, {circ[1]:.1f}), r={circ[2]:.1f}, votos={circ[3]}")
    for nombre, elip in [("Disco", hough["elipse_disco"]), ("Copa ", hough["elipse_copa"])]:
        if elip is not None:
            print(f"Hough Elipse  {nombre} -> centro=({elip[0]:.1f}, {elip[1]:.1f}), a={elip[2]:.1f}, b={elip[3]:.1f}, ang={elip[4]:.1f}°, votos={elip[5]}")


def main():
    # 1. Cargar imagen y Ground Truth
    nombre_img, img_rgb, gt_disco, gt_copa = cargarImagenYgt(RUTA_IMAGEN)

    # 2. Paso 1: Localizar disco y recortar ROI
    loc = localizarYrecortarRoi(img_rgb, gt_disco, gt_copa, SIGMA_GAUSS, TAM_ROI)

    # 3. Paso 2: Segmentación Otsu en 2 etapas dentro de la ROI
    otsu = segmentarOtsuDosEtapas(loc["roi_r"], loc["roi_g"], TAM_CIERRE)

    # 4. Paso 3: Bordes binarios + Transformada de Hough (círculos y elipses)
    hough = ajustarHough(otsu["cierre_r"], otsu["cierre_g"])

    # 5. Mostrar resumen por consola y construir figuras
    imprimirResumen(loc, otsu, hough)
    figuras = crearFiguras(img_rgb, loc, otsu, hough, SIGMA_GAUSS)

    # 6. Guardar resultados antes de mostrar las ventanas
    guardarEjecucion(figuras, params={
        "imagen": nombre_img,
        "metodo_tag": METODO_TAG,
        "metodo": "Otsu 2 etapas en ROI + Hough Círculos y Elipses sobre bordes binarios",
        "sigma_gauss": SIGMA_GAUSS,
        "umbral_otsu_r": otsu["umbral_r"],
        "umbral_otsu_g": otsu["umbral_g"],
        "cierre": TAM_CIERRE,
        "roi": TAM_ROI,
        "centroide": loc["centro"],
        "circulo_disco": tuple(round(v, 1) for v in hough["circulo_disco"]) if hough["circulo_disco"] else None,
        "circulo_copa": tuple(round(v, 1) for v in hough["circulo_copa"]) if hough["circulo_copa"] else None,
        "elipse_disco": tuple(round(v, 1) for v in hough["elipse_disco"]) if hough["elipse_disco"] else None,
        "elipse_copa": tuple(round(v, 1) for v in hough["elipse_copa"]) if hough["elipse_copa"] else None,
    })

    plt.show()


if __name__ == "__main__":
    main()
