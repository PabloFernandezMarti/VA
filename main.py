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


def guardarImagenesEjecucion(fig1, fig2, fig3, params=None, carpeta_base="ejecuciones"):
    carpeta = carpetaEjecucion(params, carpeta_base)

    fig1.savefig(os.path.join(carpeta, "1_original_y_canales.png"), dpi=150, bbox_inches="tight")
    fig2.savefig(os.path.join(carpeta, "2_mascaras_y_cierres.png"), dpi=150, bbox_inches="tight")
    fig3.savefig(os.path.join(carpeta, "3_roi_y_comparacion.png"), dpi=150, bbox_inches="tight")

    if params:
        with open(os.path.join(carpeta, "params.txt"), "w") as f:
            for clave, valor in params.items():
                f.write(f"{clave}: {valor}\n")

    print(f"Ejecución guardada en: {carpeta}")


# Cargar imagen original
ruta = './VA_P2_Public_Dataset/Images/g0023.jpg'
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

# Guardar resultados antes de mostrar las ventanas
guardarImagenesEjecucion(fig1, fig2, fig3, params={
    "imagen": nombre_img,
    "metodo_tag": "otsu2",
    "metodo": "Otsu 2 etapas en ROI (R sin borde negro -> G dentro del disco)",
    "sigma_gauss": sigma_gauss,
    "umbral_otsu_r": float(umbral_r),
    "umbral_otsu_g": float(umbral_g),
    "cierre": tam_cierre,
    "roi": tam_roi,
    "centroide": (cx, cy),
})

plt.show()