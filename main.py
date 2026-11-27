import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from datetime import datetime


def carpetaEjecucion(carpeta_base="ejecuciones"):
    fechaHora = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    ruta = os.path.join(carpeta_base, fechaHora)
    os.makedirs(ruta, exist_ok=True)
    return ruta


def guardarImagenesEjecucion(fig1, fig2, fig3, params=None, carpeta_base="ejecuciones"):
    carpeta = carpetaEjecucion(carpeta_base)

    fig1.savefig(os.path.join(carpeta, "1_original_y_canales.png"), dpi=150, bbox_inches="tight")
    fig2.savefig(os.path.join(carpeta, "2_mascaras_y_cierres.png"), dpi=150, bbox_inches="tight")
    fig3.savefig(os.path.join(carpeta, "3_roi_y_comparacion.png"), dpi=150, bbox_inches="tight")

    if params:
        with open(os.path.join(carpeta, "params.txt"), "w") as f:
            for clave, valor in params.items():
                f.write(f"{clave}: {valor}\n")

    print(f"Ejecución guardada en: {carpeta}")


# Cargar imagen original
ruta = './VA_P2_Public_Dataset/Images/n0022.jpg'
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


# --- Máscara y cierre del canal R ---
k_r = 2.5
mediana_r = np.median(r)
desviacion_r = np.std(r)
umbral_r = mediana_r + k_r * desviacion_r
mascara_r = (r >= umbral_r).astype(np.uint8)

tam_cierre = 50
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tam_cierre, tam_cierre))
cierre_r = cv2.morphologyEx(mascara_r, cv2.MORPH_CLOSE, kernel)


# --- Máscara y cierre del canal G ---
k_g = 5.0
mediana_g = np.median(g)
desviacion_g = np.std(g)
umbral_g = mediana_g + k_g * desviacion_g
mascara_g = (g >= umbral_g).astype(np.uint8)
cierre_g = cv2.morphologyEx(mascara_g, cv2.MORPH_CLOSE, kernel)

# Calcular centroide usando cierre_g y recortar ROI cuadrada
h, w = img_rgb.shape[:2]
M = cv2.moments(cierre_g)
if M["m00"] != 0:
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
else:
    cx, cy = w // 2, h // 2

tam_roi = 600
mitad = tam_roi // 2
x1 = max(0, min(cx - mitad, w - tam_roi))
y1 = max(0, min(cy - mitad, h - tam_roi))
x2 = min(w, x1 + tam_roi)
y2 = min(h, y1 + tam_roi)

roi_rgb = img_rgb[y1:y2, x1:x2]
roi_cierre_r = cierre_r[y1:y2, x1:x2]
roi_cierre_g = cierre_g[y1:y2, x1:x2]

print(f"Canal R -> Mediana: {mediana_r:.2f} | Desv.: {desviacion_r:.2f} | Umbral: {umbral_r:.2f}")
print(f"Canal G -> Mediana: {mediana_g:.2f} | Desv.: {desviacion_g:.2f} | Umbral: {umbral_g:.2f}")
print(f"Centroide: ({cx}, {cy}) | ROI: x=[{x1}, {x2}], y=[{y1}, {y2}]")

# ==============================================================================
# Ventana 1: Original, canales R, G, B y combinación (G + B) / 2
# ==============================================================================
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

axes1[5].set_visible(False)

for ax in axes1[:5]:
    ax.axis('off')
fig1.tight_layout()

# ==============================================================================
# Ventana 2: Canal origen (grises), máscara binaria (B/N) y cierre para R y G
# ==============================================================================
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9), num='Máscaras y cierres')
axes2 = axes2.ravel()

# Fila 1: Canal R
axes2[0].imshow(r, cmap='gray')
axes2[0].set_title('Canal R (Origen)')

axes2[1].imshow(mascara_r, cmap='gray', vmin=0, vmax=1)
axes2[1].set_title(f'Máscara Canal R (k={k_r})')

axes2[2].imshow(cierre_r, cmap='gray', vmin=0, vmax=1)
axes2[2].set_title('Cierre Canal R')

# Fila 2: Canal G
axes2[3].imshow(g, cmap='gray')
axes2[3].set_title('Canal G (Origen)')

axes2[4].imshow(mascara_g, cmap='gray', vmin=0, vmax=1)
axes2[4].set_title(f'Máscara Canal G (k={k_g})')

axes2[5].imshow(cierre_g, cmap='gray', vmin=0, vmax=1)
axes2[5].set_title('Cierre Canal G')

for ax in axes2:
    ax.axis('off')
fig2.tight_layout()

# ==============================================================================
# Ventana 3: ROI, máscaras ampliadas con ROI y comparación de bordes (en fila)
# ==============================================================================
contornos_pred_r, _ = cv2.findContours(cierre_r, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contornos_pred_g, _ = cv2.findContours(cierre_g, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contornos_gt_disco, _ = cv2.findContours(gt_disco, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornos_gt_copa, _ = cv2.findContours(gt_copa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

comparacion_bordes = img_rgb.copy()
cv2.drawContours(comparacion_bordes, contornos_gt_disco, -1, (255, 0, 255), 3)  # magenta: ref. disco
cv2.drawContours(comparacion_bordes, contornos_gt_copa, -1, (0, 255, 0), 3)     # verde: ref. copa
cv2.drawContours(comparacion_bordes, contornos_pred_r, -1, (255, 255, 0), 3)    # amarillo: pred. R
cv2.drawContours(comparacion_bordes, contornos_pred_g, -1, (0, 255, 255), 3)    # cian: pred. G
roi_comparacion = comparacion_bordes[y1:y2, x1:x2]

fig3, axes3 = plt.subplots(1, 4, figsize=(22, 6), num='ROI y comparación')

axes3[0].imshow(roi_rgb)
axes3[0].set_title('ROI - Original')

axes3[1].imshow(roi_cierre_r, cmap='gray', vmin=0, vmax=1)
axes3[1].set_title('ROI - Máscara R (Cierre)')

axes3[2].imshow(roi_cierre_g, cmap='gray', vmin=0, vmax=1)
axes3[2].set_title('ROI - Máscara G (Cierre)')

axes3[3].imshow(roi_comparacion)
axes3[3].set_title('Comparación de bordes (ROI)')

leyenda = [
    Line2D([0], [0], color=(1, 1, 0), lw=2, label='Pred. Canal R'),
    Line2D([0], [0], color=(0, 1, 1), lw=2, label='Pred. Canal G'),
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
    "k_r": k_r,
    "k_g": k_g,
    "cierre": tam_cierre,
    "roi": tam_roi,
    "centroide": (cx, cy),
})

plt.show()