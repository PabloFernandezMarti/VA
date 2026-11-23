import cv2
import numpy as np
import matplotlib.pyplot as plt

ruta = './VA_P2_Public_Dataset/Images/g0001.jpg'
img_bgr = cv2.imread(ruta)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# separar rgb
r, g, b = cv2.split(img_rgb)

# combinar canales G y B
gb = (g.astype(np.float32) + b.astype(np.float32)) / 2.0


# umbral mediante mediana mas desviacion entre 2.0 y 4.0)
k = 4.0
mediana = np.median(gb)
desviacion = np.std(gb)
umbral = mediana + k * desviacion

# umbral para sacar mascara de la copa
mascara_copa = (gb >= umbral).astype(np.uint8)

# cierre para quitar venas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
mascara_cierre = cv2.morphologyEx(mascara_copa, cv2.MORPH_CLOSE, kernel)

print(f"Mediana: {mediana:.2f} | Desv. típica: {desviacion:.2f} | Umbral: {umbral:.2f}")

# mostrar resultados
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
axes = axes.ravel()

axes[0].imshow(img_rgb)
axes[0].set_title('Original (RGB)')

axes[1].imshow(r, cmap='gray')
axes[1].set_title('Canal R')

axes[2].imshow(g, cmap='gray')
axes[2].set_title('Canal G')

axes[3].imshow(b, cmap='gray')
axes[3].set_title('Canal B')

axes[4].imshow(gb, cmap='gray')
axes[4].set_title('Combinación (G + B) / 2')

axes[5].imshow(mascara_copa, cmap='gray')
axes[5].set_title(f'Máscara Copa (k={k})')

axes[6].imshow(mascara_cierre, cmap='gray')
axes[6].set_title('Máscara tras Cierre')

for ax in axes:
    ax.axis('off')

plt.tight_layout()
plt.show()