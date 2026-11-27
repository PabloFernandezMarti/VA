import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


def carpetaEjecucion(params=None, carpeta_base="ejecuciones"):
    partes = []
    if params:
        if "imagen" in params:
            partes.append(str(params["imagen"]))
        if "metodo_tag" in params:
            partes.append(str(params["metodo_tag"]))

    nombre_carpeta = "_".join(partes) if partes else "ejecucion"
    ruta = os.path.join(carpeta_base, nombre_carpeta)
    os.makedirs(ruta, exist_ok=True)
    return ruta


def guardarImagenesEjecucion(fig1, fig2, fig3, fig4=None, fig5=None, params=None, carpeta_base="ejecuciones"):
    carpeta = carpetaEjecucion(params, carpeta_base)

    fig1.savefig(os.path.join(carpeta, "1_original_y_canales.png"), dpi=150, bbox_inches="tight")
    fig2.savefig(os.path.join(carpeta, "2_mascaras_y_cierres.png"), dpi=150, bbox_inches="tight")
    fig3.savefig(os.path.join(carpeta, "3_roi_y_comparacion.png"), dpi=150, bbox_inches="tight")
    if fig4 is not None:
        fig4.savefig(os.path.join(carpeta, "4_bordes_y_hough.png"), dpi=150, bbox_inches="tight")
    if fig5 is not None:
        fig5.savefig(os.path.join(carpeta, "5_hough_vs_gt.png"), dpi=150, bbox_inches="tight")

    if params:
        with open(os.path.join(carpeta, "params.txt"), "w") as f:
            for clave, valor in params.items():
                f.write(f"{clave}: {valor}\n")

    print(f"Ejecución guardada en: {carpeta}")


def extraerBordesBinarios(mascara_bin):
    """Extrae el borde binario (1 píxel) directamente de una máscara binaria (0/1)."""
    kernel_3 = np.ones((3, 3), np.uint8)
    return (mascara_bin - cv2.erode(mascara_bin, kernel_3)).astype(np.uint8)


def houghElipse(bordes_bin, escala=0.15, min_ratio=0.6, paso_b=1.5):
    """
    Transformada de Hough para elipses (Xie & Ji, 2002) sobre una imagen de bordes binaria.
    Devuelve (xc, yc, a, b, angulo_grados, votos) en coordenadas de la imagen original.
    """
    ys, xs = np.nonzero(bordes_bin)
    if xs.size < 10:
        return None

    # Reducir resolución de los puntos de borde para acelerar el acumulador O(N^3)
    pts = np.unique(np.column_stack((np.round(xs * escala), np.round(ys * escala))), axis=0)
    if len(pts) > 120:
        idx = np.linspace(0, len(pts) - 1, 120, dtype=int)
        pts = pts[idx]

    # Acotar el eje mayor (2a) según el tamaño aproximado del borde detectado
    span = max(pts[:, 0].max() - pts[:, 0].min(), pts[:, 1].max() - pts[:, 1].min())
    min_2a = max(6.0, span * 0.55)
    max_2a = span * 1.15

    px = pts[:, 0]
    py = pts[:, 1]
    n = len(pts)
    mejor_votos = -1
    mejor_elipse = None

    for i in range(n - 1):
        dx = px[i + 1:] - px[i]
        dy = py[i + 1:] - py[i]
        dist = np.hypot(dx, dy)

        for off in np.where((dist >= min_2a) & (dist <= max_2a))[0]:
            j = i + 1 + off
            a = 0.5 * dist[off]
            x0 = 0.5 * (px[i] + px[j])
            y0 = 0.5 * (py[i] + py[j])
            alpha = np.arctan2(py[j] - py[i], px[j] - px[i])
            ca, sa = np.cos(alpha), np.sin(alpha)

            # Proyectar el resto de puntos de borde sobre los ejes de la elipse candidata
            rx = px - x0
            ry = py - y0
            u = rx * ca + ry * sa
            v = -rx * sa + ry * ca

            u2_a2 = (u * u) / (a * a)
            validos = u2_a2 < 0.99
            if not np.any(validos):
                continue

            # Despejar el semieje menor b y votar en el acumulador 1D
            b_vals = np.sqrt((v[validos] * v[validos]) / (1.0 - u2_a2[validos]))
            b_valid = b_vals[(b_vals >= min_ratio * a) & (b_vals <= a * 1.05)]
            if b_valid.size == 0:
                continue

            bins = np.floor(b_valid / paso_b).astype(np.int32)
            counts = np.bincount(bins)
            idx_max = int(np.argmax(counts))
            votos = int(counts[idx_max])

            if votos > mejor_votos:
                mejor_votos = votos
                b_est = (idx_max + 0.5) * paso_b
                mejor_elipse = (
                    x0 / escala,
                    y0 / escala,
                    a / escala,
                    b_est / escala,
                    float(np.degrees(alpha)),
                    votos,
                )

    return mejor_elipse


def dibujarElipse(imagen, elipse, color, grosor=-1):
    """Dibuja una elipse (xc, yc, a, b, angulo, votos) sobre imagen con cv2.ellipse."""
    if elipse is None:
        return
    xc, yc, a, b, angulo, _ = elipse
    centro = (int(round(xc)), int(round(yc)))
    ejes = (max(1, int(round(a))), max(1, int(round(b))))
    cv2.ellipse(imagen, centro, ejes, angulo, 0, 360, color, grosor)


def houghCirculo(bordes_bin, escala=0.2):
    """
    Transformada de Hough estándar para círculos (xc, yc, r) sobre bordes binarios.
    Devuelve (xc, yc, r, votos) en coordenadas de la imagen original.
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

        idx = int(np.argmax(acc))
        cy_m, cx_m = divmod(idx, nw)
        votos = int(acc[cy_m, cx_m])

        if votos > mejor_votos:
            mejor_votos = votos
            mejor_circulo = (cx_m / escala, cy_m / escala, r / escala, votos)

    return mejor_circulo


def dibujarCirculo(imagen, circulo, color, grosor=-1):
    """Dibuja un círculo (xc, yc, r, votos) sobre imagen con cv2.circle."""
    if circulo is None:
        return
    xc, yc, r, _ = circulo
    centro = (int(round(xc)), int(round(yc)))
    radio = max(1, int(round(r)))
    cv2.circle(imagen, centro, radio, color, grosor)


# Cargar imagen original
ruta = './VA_P2_Public_Dataset/Images/n0023.jpg'
img_bgr = cv2.imread(ruta)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Cargar máscara de referencia (ground truth: 0=fondo, 1=disco/rim, 2=copa)
nombre_img = os.path.splitext(os.path.basename(ruta))[0]
ruta_gt = f'./VA_P2_Public_Dataset/Masks/{nombre_img}.png'
gt = cv2.imread(ruta_gt, cv2.IMREAD_GRAYSCALE)
gt_disco = (gt >= 1).astype(np.uint8)
gt_copa = (gt == 2).astype(np.uint8)

# Separar canales RGB y combinar G y B
r, g, b = cv2.split(img_rgb)
gb = (g.astype(np.float32) + b.astype(np.float32)) / 2.0


# --- Paso 1: Localizar centro del disco mediante suavizado Gaussiano en G y recortar ROI ---
sigma_gauss = 50
g_suavizado = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma_gauss)
_, _, _, max_loc = cv2.minMaxLoc(g_suavizado)
cx, cy = max_loc

h, w = img_rgb.shape[:2]
tam_roi = 600
mitad = tam_roi // 2
x1 = max(0, min(cx - mitad, w - tam_roi))
y1 = max(0, min(cy - mitad, h - tam_roi))
x2 = min(w, x1 + tam_roi)
y2 = min(h, y1 + tam_roi)

roi_rgb = img_rgb[y1:y2, x1:x2]
roi_r = r[y1:y2, x1:x2]
roi_g = g[y1:y2, x1:x2]
roi_gt_disco = gt_disco[y1:y2, x1:x2]
roi_gt_copa = gt_copa[y1:y2, x1:x2]


# --- Paso 2: Umbralización de Otsu en 2 etapas y cierre en la ROI ---
tam_cierre = 50
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam_cierre, tam_cierre))

# Etapa 1: Otsu en canal R (ROI) ignorando el borde negro exterior del ojo para separar el disco
pixeles_r_validos = roi_r[roi_r > 10]
if pixeles_r_validos.size > 0:
    umbral_r, _ = cv2.threshold(pixeles_r_validos.reshape(-1, 1), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
else:
    umbral_r, _ = cv2.threshold(roi_r, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
mascara_r = (roi_r > umbral_r).astype(np.uint8)
cierre_r = cv2.morphologyEx(mascara_r, cv2.MORPH_CLOSE, kernel)

# Etapa 2: Otsu en canal G restringido solo a los píxeles dentro del disco (cierre_r == 1) para separar rim vs. copa
roi_g_disco = roi_g * cierre_r
pixeles_g_disco = roi_g[cierre_r == 1]
if pixeles_g_disco.size > 0:
    umbral_g, _ = cv2.threshold(pixeles_g_disco.reshape(-1, 1), 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
else:
    umbral_g, _ = cv2.threshold(roi_g, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
mascara_g = ((roi_g > umbral_g) & (cierre_r == 1)).astype(np.uint8)
cierre_g = cv2.morphologyEx(mascara_g, cv2.MORPH_CLOSE, kernel)

print(f"Centroide (Gauss sigma={sigma_gauss}): ({cx}, {cy}) | ROI: x=[{x1}, {x2}], y=[{y1}, {y2}]")
print(f"Otsu Canal R (Disco, R>10)    -> Umbral: {umbral_r:.0f}")
print(f"Otsu Canal G (Copa, en disco) -> Umbral: {umbral_g:.0f}")

# Ventana 1: Original, canales R, G, B, combinación (G + B) / 2 y localización ROI
fig1, axes1 = plt.subplots(2, 3, figsize=(15, 9), num='Original y canales')
axes1 = axes1.ravel()

axes1[0].imshow(img_rgb)
axes1[0].set_title('Original (RGB)')

axes1[1].imshow(r, cmap='gray')
axes1[1].set_title('Canal R')

axes1[2].imshow(g, cmap='gray')
axes1[2].set_title('Canal G')

axes1[3].imshow(b, cmap='gray')
axes1[3].set_title('Canal B')

axes1[4].imshow(gb, cmap='gray')
axes1[4].set_title('Combinación (G + B) / 2')

img_roi_loc = img_rgb.copy()
cv2.rectangle(img_roi_loc, (x1, y1), (x2, y2), (0, 255, 255), 8)
cv2.circle(img_roi_loc, (cx, cy), 15, (255, 0, 0), -1)
axes1[5].imshow(img_roi_loc)
axes1[5].set_title(f'Localización ROI (Gauss σ={sigma_gauss})')

for ax in axes1:
    ax.axis('off')
fig1.tight_layout()

# Ventana 2: Canal ROI (grises), máscara Otsu (B/N) y cierre para R (disco) y G (copa)
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9), num='Máscaras Otsu y cierres (ROI)')
axes2 = axes2.ravel()

# Fila 1: Canal R (Disco)
axes2[0].imshow(roi_r, cmap='gray')
axes2[0].set_title('ROI - Canal R (Disco)')

axes2[1].imshow(mascara_r, cmap='gray', vmin=0, vmax=1)
axes2[1].set_title(f'Otsu Canal R (umbral={umbral_r:.0f})')

axes2[2].imshow(cierre_r, cmap='gray', vmin=0, vmax=1)
axes2[2].set_title('Cierre Canal R (Disco)')

# Fila 2: Canal G restringido al disco (Copa)
axes2[3].imshow(roi_g_disco, cmap='gray')
axes2[3].set_title('ROI - Canal G (dentro del disco)')

axes2[4].imshow(mascara_g, cmap='gray', vmin=0, vmax=1)
axes2[4].set_title(f'Otsu Canal G en disco (umbral={umbral_g:.0f})')

axes2[5].imshow(cierre_g, cmap='gray', vmin=0, vmax=1)
axes2[5].set_title('Cierre Canal G (Copa)')

for ax in axes2:
    ax.axis('off')
fig2.tight_layout()

# Ventana 3: ROI, máscaras Otsu con cierre y comparación de bordes (en fila)
contornos_pred_r, _ = cv2.findContours(cierre_r, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contornos_pred_g, _ = cv2.findContours(cierre_g, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contornos_gt_disco, _ = cv2.findContours(roi_gt_disco, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornos_gt_copa, _ = cv2.findContours(roi_gt_copa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

roi_comparacion = roi_rgb.copy()
cv2.drawContours(roi_comparacion, contornos_gt_disco, -1, (255, 0, 255), 3)  # magenta: ref. disco
cv2.drawContours(roi_comparacion, contornos_gt_copa, -1, (0, 255, 0), 3)     # verde: ref. copa
cv2.drawContours(roi_comparacion, contornos_pred_r, -1, (255, 255, 0), 3)    # amarillo: pred. disco (R)
cv2.drawContours(roi_comparacion, contornos_pred_g, -1, (0, 255, 255), 3)    # cian: pred. copa (G)

fig3, axes3 = plt.subplots(1, 4, figsize=(22, 6), num='ROI y comparación')

axes3[0].imshow(roi_rgb)
axes3[0].set_title('ROI - Original')

axes3[1].imshow(cierre_r, cmap='gray', vmin=0, vmax=1)
axes3[1].set_title('ROI - Disco Otsu (Cierre)')

axes3[2].imshow(cierre_g, cmap='gray', vmin=0, vmax=1)
axes3[2].set_title('ROI - Copa Otsu (Cierre)')

axes3[3].imshow(roi_comparacion)
axes3[3].set_title('Comparación de bordes (ROI)')

leyenda = [
    Line2D([0], [0], color=(1, 1, 0), lw=2, label='Pred. Disco (Otsu R)'),
    Line2D([0], [0], color=(0, 1, 1), lw=2, label='Pred. Copa (Otsu G)'),
    Line2D([0], [0], color=(1, 0, 1), lw=2, label='Ref. Disco'),
    Line2D([0], [0], color=(0, 1, 0), lw=2, label='Ref. Copa'),
]
axes3[3].legend(handles=leyenda, loc='upper right', fontsize=9, framealpha=0.8)

for ax in axes3:
    ax.axis('off')
fig3.tight_layout()

# --- Paso 3: Extracción de bordes binarios y Transformada de Hough (círculos y elipses) ---
bordes_r = extraerBordesBinarios(cierre_r)
bordes_g = extraerBordesBinarios(cierre_g)

circulo_disco = houghCirculo(bordes_r)
circulo_copa = houghCirculo(bordes_g)

elipse_disco = houghElipse(bordes_r)
elipse_copa = houghElipse(bordes_g)

mascara_hough_circ_r = np.zeros_like(cierre_r, dtype=np.uint8)
mascara_hough_circ_g = np.zeros_like(cierre_g, dtype=np.uint8)
dibujarCirculo(mascara_hough_circ_r, circulo_disco, 1, grosor=-1)
dibujarCirculo(mascara_hough_circ_g, circulo_copa, 1, grosor=-1)

mascara_hough_r = np.zeros_like(cierre_r, dtype=np.uint8)
mascara_hough_g = np.zeros_like(cierre_g, dtype=np.uint8)
dibujarElipse(mascara_hough_r, elipse_disco, 1, grosor=-1)
dibujarElipse(mascara_hough_g, elipse_copa, 1, grosor=-1)

if circulo_disco is not None:
    print(f"Hough Círculo Disco -> centro=({circulo_disco[0]:.1f}, {circulo_disco[1]:.1f}), r={circulo_disco[2]:.1f}, votos={circulo_disco[3]}")
if circulo_copa is not None:
    print(f"Hough Círculo Copa  -> centro=({circulo_copa[0]:.1f}, {circulo_copa[1]:.1f}), r={circulo_copa[2]:.1f}, votos={circulo_copa[3]}")
if elipse_disco is not None:
    print(f"Hough Elipse Disco  -> centro=({elipse_disco[0]:.1f}, {elipse_disco[1]:.1f}), a={elipse_disco[2]:.1f}, b={elipse_disco[3]:.1f}, ang={elipse_disco[4]:.1f}°, votos={elipse_disco[5]}")
if elipse_copa is not None:
    print(f"Hough Elipse Copa   -> centro=({elipse_copa[0]:.1f}, {elipse_copa[1]:.1f}), a={elipse_copa[2]:.1f}, b={elipse_copa[3]:.1f}, ang={elipse_copa[4]:.1f}°, votos={elipse_copa[5]}")

# Ventana 4: Bordes binarios de las máscaras de Otsu y ajuste de elipses con Hough
bordes_r_vis = cv2.cvtColor(bordes_r * 255, cv2.COLOR_GRAY2RGB)
dibujarElipse(bordes_r_vis, elipse_disco, (255, 255, 0), grosor=3)  # amarillo

bordes_g_vis = cv2.cvtColor(bordes_g * 255, cv2.COLOR_GRAY2RGB)
dibujarElipse(bordes_g_vis, elipse_copa, (0, 255, 255), grosor=3)   # cian

fig4, axes4 = plt.subplots(2, 3, figsize=(15, 9), num='Bordes binarios y Hough Elipses (ROI)')
axes4 = axes4.ravel()

# Fila 1: Disco (R)
axes4[0].imshow(bordes_r, cmap='gray', vmin=0, vmax=1)
axes4[0].set_title('Bordes binarios Disco (Otsu R)')

axes4[1].imshow(bordes_r_vis)
axes4[1].set_title('Hough Elipse sobre bordes Disco')

axes4[2].imshow(mascara_hough_r, cmap='gray', vmin=0, vmax=1)
axes4[2].set_title('Máscara Elipse Hough (Disco)')

# Fila 2: Copa (G)
axes4[3].imshow(bordes_g, cmap='gray', vmin=0, vmax=1)
axes4[3].set_title('Bordes binarios Copa (Otsu G)')

axes4[4].imshow(bordes_g_vis)
axes4[4].set_title('Hough Elipse sobre bordes Copa')

axes4[5].imshow(mascara_hough_g, cmap='gray', vmin=0, vmax=1)
axes4[5].set_title('Máscara Elipse Hough (Copa)')

for ax in axes4:
    ax.axis('off')
fig4.tight_layout()

# Ventana 5: Comparación final de Hough Círculos (fila 1) y Hough Elipses (fila 2) frente al Ground Truth
roi_gt_vis = np.zeros_like(roi_gt_disco, dtype=np.uint8)
roi_gt_vis[roi_gt_disco == 1] = 128
roi_gt_vis[roi_gt_copa == 1] = 255

hough_circ_vis = np.zeros_like(mascara_hough_circ_r, dtype=np.uint8)
hough_circ_vis[mascara_hough_circ_r == 1] = 128
hough_circ_vis[mascara_hough_circ_g == 1] = 255

roi_comparacion_circ = roi_rgb.copy()
cv2.drawContours(roi_comparacion_circ, contornos_gt_disco, -1, (255, 0, 255), 3)  # magenta: ref. disco
cv2.drawContours(roi_comparacion_circ, contornos_gt_copa, -1, (0, 255, 0), 3)     # verde: ref. copa
dibujarCirculo(roi_comparacion_circ, circulo_disco, (255, 255, 0), grosor=3)      # amarillo: Hough círculo disco
dibujarCirculo(roi_comparacion_circ, circulo_copa, (0, 255, 255), grosor=3)       # cian: Hough círculo copa

hough_vis = np.zeros_like(mascara_hough_r, dtype=np.uint8)
hough_vis[mascara_hough_r == 1] = 128
hough_vis[mascara_hough_g == 1] = 255

roi_comparacion_hough = roi_rgb.copy()
cv2.drawContours(roi_comparacion_hough, contornos_gt_disco, -1, (255, 0, 255), 3)  # magenta: ref. disco
cv2.drawContours(roi_comparacion_hough, contornos_gt_copa, -1, (0, 255, 0), 3)     # verde: ref. copa
dibujarElipse(roi_comparacion_hough, elipse_disco, (255, 255, 0), grosor=3)        # amarillo: Hough elipse disco
dibujarElipse(roi_comparacion_hough, elipse_copa, (0, 255, 255), grosor=3)         # cian: Hough elipse copa

fig5, axes5 = plt.subplots(2, 3, figsize=(18, 11), num='Hough Círculos y Elipses vs Ground Truth (ROI)')
axes5 = axes5.ravel()

# Fila 1: Hough Círculos
axes5[0].imshow(roi_gt_vis, cmap='gray', vmin=0, vmax=255)
axes5[0].set_title('Ground Truth (Disco gris, Copa blanca)')

axes5[1].imshow(hough_circ_vis, cmap='gray', vmin=0, vmax=255)
axes5[1].set_title('Hough Círculos (Disco gris, Copa blanca)')

axes5[2].imshow(roi_comparacion_circ)
axes5[2].set_title('Comparación de bordes: Hough Círculos vs GT')

leyenda_circ = [
    Line2D([0], [0], color=(1, 1, 0), lw=2, label='Hough Círculo Disco'),
    Line2D([0], [0], color=(0, 1, 1), lw=2, label='Hough Círculo Copa'),
    Line2D([0], [0], color=(1, 0, 1), lw=2, label='Ref. Disco'),
    Line2D([0], [0], color=(0, 1, 0), lw=2, label='Ref. Copa'),
]
axes5[2].legend(handles=leyenda_circ, loc='upper right', fontsize=9, framealpha=0.8)

# Fila 2: Hough Elipses
axes5[3].imshow(roi_gt_vis, cmap='gray', vmin=0, vmax=255)
axes5[3].set_title('Ground Truth (Disco gris, Copa blanca)')

axes5[4].imshow(hough_vis, cmap='gray', vmin=0, vmax=255)
axes5[4].set_title('Hough Elipses (Disco gris, Copa blanca)')

axes5[5].imshow(roi_comparacion_hough)
axes5[5].set_title('Comparación de bordes: Hough Elipses vs GT')

leyenda_hough = [
    Line2D([0], [0], color=(1, 1, 0), lw=2, label='Hough Elipse Disco'),
    Line2D([0], [0], color=(0, 1, 1), lw=2, label='Hough Elipse Copa'),
    Line2D([0], [0], color=(1, 0, 1), lw=2, label='Ref. Disco'),
    Line2D([0], [0], color=(0, 1, 0), lw=2, label='Ref. Copa'),
]
axes5[5].legend(handles=leyenda_hough, loc='upper right', fontsize=9, framealpha=0.8)

for ax in axes5:
    ax.axis('off')
fig5.tight_layout()

# Guardar resultados antes de mostrar las ventanas
guardarImagenesEjecucion(fig1, fig2, fig3, fig4=fig4, fig5=fig5, params={
    "imagen": nombre_img,
    "metodo_tag": "otsu2_hough",
    "metodo": "Otsu 2 etapas en ROI + Hough Círculos y Elipses sobre bordes binarios",
    "sigma_gauss": sigma_gauss,
    "umbral_otsu_r": float(umbral_r),
    "umbral_otsu_g": float(umbral_g),
    "cierre": tam_cierre,
    "roi": tam_roi,
    "centroide": (cx, cy),
    "circulo_disco": tuple(round(v, 1) for v in circulo_disco) if circulo_disco else None,
    "circulo_copa": tuple(round(v, 1) for v in circulo_copa) if circulo_copa else None,
    "elipse_disco": tuple(round(v, 1) for v in elipse_disco) if elipse_disco else None,
    "elipse_copa": tuple(round(v, 1) for v in elipse_copa) if elipse_copa else None,
})

plt.show()