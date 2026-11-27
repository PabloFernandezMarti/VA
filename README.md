# Práctica 2 — Visión Artificial (Segmentación de disco óptico y copa)

Registro de la evolución del método desarrollado en `main.py` para la segmentación del disco óptico y la copa en retinografías (dataset REFUGE), pensado como base de trabajo para la memoria de la práctica.

---

## 1. Exploración inicial: análisis por canales

Al comienzo no teníamos claro cómo abordar la segmentación, así que empezamos descomponiendo las retinografías en sus tres canales (**R**, **G** y **B**) y probando combinaciones como `(G + B) / 2` para observar cómo se distribuía la información en cada uno.

De esta exploración sacamos dos conclusiones clave:
- Las estructuras de interés (disco y copa) se corresponden con las **regiones más claras** de la imagen.
- Cada canal aporta información distinta:
  - El **canal R (rojo)** separa mejor el **disco óptico** (el círculo completo) respecto al fondo de la retina.
  - El **canal G (verde)** ofrece mayor contraste para diferenciar la **copa óptica** del anillo neuroretiniano (*rim*).

---

## 2. Primera aproximación: umbralización por mediana + desviación y cierre morfológico

Con las observaciones anteriores planteamos un primer método basado en dos máscaras independientes:

1. **Umbralización estadística (`mediana + k · desviación típica`):**
   - Una máscara sobre el **canal R** para extraer el disco (`k_r = 2.5`).
   - Otra máscara sobre el **canal G** para extraer la copa (`k_g = 5.0`).
2. **Problema encontrado — ruido por vasos sanguíneos:**
   - Al aplicar el umbral directamente, las venas que atraviesan el disco y la copa aparecían oscuras, fragmentando las máscaras y dejando huecos en su interior.
3. **Solución — operación morfológica de cierre:**
   - Aplicamos un **cierre morfológico** (`cv2.MORPH_CLOSE`) con un elemento estructurante elíptico (`50×50`) sobre ambas máscaras para rellenar los huecos causados por los vasos y conseguir formas más homogéneas.
4. **Cálculo inicial de la ROI:**
   - En esta fase la región de interés (ROI cuadrada de `600×600`) se obtenía calculando el **centroide de la máscara** tras el cierre (`cv2.moments`) y recortando alrededor de ese punto para visualizar y comparar con el *ground truth*.

### Limitación detectada
Aunque ajustando los factores `k_r` y `k_g` el resultado parecía funcionar bien en imágenes concretas, pronto comprobamos que **no era generalizable**: los cambios de iluminación y contraste entre distintas retinografías hacían que unos mismos valores fijos de `k` sobrestimasen o infraestimasen las máscaras en otras imágenes.

---

## 3. Situación actual: pruebas con Otsu y nueva localización de la ROI

Para evitar depender de umbrales ajustados a mano, actualmente estamos probando la **umbralización automática mediante el algoritmo de Otsu**.

### Cambio en la detección de la ROI
Como Otsu divide la imagen en dos clases, aplicarlo sobre la retinografía completa separa el borde negro exterior del resto del ojo, en lugar de separar disco y copa. Por ello se invirtió el orden del flujo y se cambió cómo se calcula la ROI:
- **Antes:** centroide de la máscara umbralizada (`cv2.moments`).
- **Ahora:** localización directa del **punto más brillante** sobre el canal `G` tras aplicar un suavizado Gaussiano (`sigma = 50`) para evitar falsos máximos puntuales, recortando ahí la ROI de `600×600` antes de umbralizar.

### Otsu en dos etapas dentro de la ROI
1. **Primer intento (Otsu independiente en toda la ROI):**
   - Al aplicar Otsu directamente sobre toda la ROI en el canal `G`, la máscara de la copa captaba demasiada región exterior (parte del *rim*). Esto ocurre porque en la ROI hay tres clases (fondo de retina, *rim* y copa), y al dominar el fondo oscuro, Otsu colocaba el umbral entre el fondo y el disco entero en lugar de separar *rim* y copa.
2. **Enfoque actual en pruebas (Otsu jerárquico en 2 etapas):**
   - **Etapa 1 — Disco (canal R en la ROI):** Otsu sobre los píxeles válidos de `roi_r` (`roi_r > 10`, excluyendo posibles bordes negros del exterior del ojo) seguido de cierre morfológico para obtener la máscara del disco.
   - **Etapa 2 — Copa (canal G dentro del disco):** Otsu restringido únicamente a los píxeles de `roi_g` que pertenecen al disco segmentado en la etapa 1, de modo que solo queden dos clases (*rim* vs. *copa*), seguido de cierre morfológico.

---

## 4. Seguimiento de pruebas y próximos pasos

- **Registro de ejecuciones:** cada ejecución de `main.py` guarda automáticamente en `ejecuciones/<imagen>_<parametros>/` las tres figuras generadas (canales y localización de ROI, máscaras/cierres, y comparación de contornos frente al *ground truth*) junto con `params.txt`.
- **Pendiente de abordar:**
  - Validar el enfoque de Otsu en 2 etapas sobre el conjunto completo de imágenes de desarrollo.
  - Implementar métricas cuantitativas frente al *ground truth* (localización de disco, segmentación de disco/copa y cálculo del CDR).
  - Evaluar posibles refinamientos (p. ej. suavizado de contornos, selección de la componente conexa principal o ajuste a elipses) a partir de los errores observados.
