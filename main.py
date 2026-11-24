import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ruta = './VA_P2_Public_Dataset/Images/n0013.jpg'
img_bgr = cv2.imread(ruta)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# cargar máscara de referencia (ground truth: 0=fondo, 1=disco/rim, 2=copa)
nombre_img = os.path.splitext(os.path.basename(ruta))[0]
ruta_gt = f'./VA_P2_Public_Dataset/Masks/{nombre_img}.png'
gt = cv2.imread(ruta_gt, cv2.IMREAD_GRAYSCALE)
gt_disco = (gt >= 1).astype(np.uint8)
gt_copa = (gt == 2).astype(np.uint8)

# separar rgb
r, g, b = cv2.split(img_rgb)

# combinar canales G y B
gb = (g.astype(np.float32) + b.astype(np.float32)) / 2.0


# umbral mediante mediana mas desviacion entre 2.0 y 4.0)
k = 5.0
mediana = np.median(gb)
desviacion = np.std(gb)
umbral = mediana + k * desviacion

# umbral para sacar mascara de la copa
mascara_copa = (gb >= umbral).astype(np.uint8)

# cierre para quitar venas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
mascara_cierre = cv2.morphologyEx(mascara_copa, cv2.MORPH_CLOSE, kernel)

# calcular centroide de la máscara tras el cierre y recortar ROI cuadrada
h, w = img_rgb.shape[:2]
M = cv2.moments(mascara_cierre)
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
roi_r = r[y1:y2, x1:x2]
roi_g = g[y1:y2, x1:x2]
roi_b = b[y1:y2, x1:x2]
roi_gb = gb[y1:y2, x1:x2]

print(f"Mediana: {mediana:.2f} | Desv. típica: {desviacion:.2f} | Umbral: {umbral:.2f}")
print(f"Centroide: ({cx}, {cy}) | ROI: x=[{x1}, {x2}], y=[{y1}, {y2}]")

# Ventana 1: Imagen original, canales R, G, B y media (G + B) / 2
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

# Ventana 2: ROI sobre la imagen original y los 4 canales
fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9), num='ROI sobre original y canales')
axes2 = axes2.ravel()

axes2[0].imshow(roi_rgb)
axes2[0].set_title('ROI - Original (RGB)')

axes2[1].imshow(roi_r, cmap='gray')
axes2[1].set_title('ROI - Canal R')

axes2[2].imshow(roi_g, cmap='gray')
axes2[2].set_title('ROI - Canal G')

axes2[3].imshow(roi_b, cmap='gray')
axes2[3].set_title('ROI - Canal B')

axes2[4].imshow(roi_gb, cmap='gray')
axes2[4].set_title('ROI - Combinación (G + B) / 2')

axes2[5].set_visible(False)

for ax in axes2[:5]:
    ax.axis('off')
fig2.tight_layout()

# extraer contornos de nuestra máscara y de la máscara de referencia
contornos_pred, _ = cv2.findContours(mascara_cierre, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
contornos_gt_copa, _ = cv2.findContours(gt_copa, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contornos_gt_disco, _ = cv2.findContours(gt_disco, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# dibujar bordes sobre la imagen original y recortar a la ROI para comparar
comparacion_bordes = img_rgb.copy()
cv2.drawContours(comparacion_bordes, contornos_gt_disco, -1, (255, 0, 255), 3)  # magenta: ref. disco
cv2.drawContours(comparacion_bordes, contornos_gt_copa, -1, (0, 255, 0), 3)     # verde: ref. copa
cv2.drawContours(comparacion_bordes, contornos_pred, -1, (0, 255, 255), 3)      # cian: nuestra máscara
roi_comparacion = comparacion_bordes[y1:y2, x1:x2]

# Ventana 3: Máscara, operaciones y comparación de bordes
fig3, axes3 = plt.subplots(1, 3, figsize=(18, 6), num='Máscara y operaciones')

axes3[0].imshow(mascara_copa, cmap='gray')
axes3[0].set_title(f'Máscara Copa (k={k})')

axes3[1].imshow(mascara_cierre, cmap='gray')
axes3[1].set_title('Máscara tras Cierre')

axes3[2].imshow(roi_comparacion)
axes3[2].set_title('Comparación de bordes (ROI)')
leyenda = [
    Line2D([0], [0], color=(0, 1, 1), lw=2, label='Nuestra máscara'),
    Line2D([0], [0], color=(0, 1, 0), lw=2, label='Ref. Copa'),
    Line2D([0], [0], color=(1, 0, 1), lw=2, label='Ref. Disco'),
]
axes3[2].legend(handles=leyenda, loc='upper right', fontsize=9, framealpha=0.8)

for ax in axes3:
    ax.axis('off')
fig3.tight_layout()

plt.show()