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

## 3. Umbralización automática con Otsu y nueva localización de la ROI

Para evitar depender de umbrales ajustados a mano, pasamos a usar la **umbralización automática mediante el algoritmo de Otsu**.

### Cambio en la detección de la ROI y refinamiento de canales (`G` → `GB` → `GR`)
Como Otsu divide la imagen en dos clases, aplicarlo sobre la retinografía completa separa el borde negro exterior del resto del ojo, en lugar de separar disco y copa. Por ello se invirtió el orden del flujo y se cambió cómo se calcula la ROI:
- **Antes:** centroide de la máscara umbralizada (`cv2.moments`).
- **Ahora:** localización directa del **punto más brillante** tras aplicar un suavizado Gaussiano fuerte (`sigma = 50`) para evitar falsos máximos puntuales, recortando ahí la ROI cuadrada de `600×600` antes de umbralizar.

Al evaluar esta localización sobre las **40 imágenes de desarrollo (`dev`)**, probamos distintos canales de entrada y observamos una limitación clara que nos llevó a refinar la combinación final:
- **Canal `G` solo:** acierta en `39/40` imágenes (`97.5 %`), pero falla en **`n0230`** (error de `1305.7 px`, cobertura `0 %`), porque esa retinografía tiene un **halo verdoso muy brillante en el borde inferior del campo de visión (FOV)** que supera al disco tras el suavizado.
- **Combinación `(G + B) / 2`:** mantiene `39/40` (`97.5 %`) y reduce ligeramente el error mediano (`52.0 px`), pero **sigue fallando en `n0230`** porque ese halo del borde inferior también tiene intensidad en el canal azul (`B`).
- **Combinación `(G + R) / 2` (solución adoptada):** logra **`40/40` (`100.0 %`)** de ROIs conteniendo el disco completo, con un error de centroide medio de `47.6 px` (mediana `51.9 px`, máximo `77.0 px`).
  - *Motivo físico:* usar solo `R` puede confundirse con zonas rojizas difusas de la retina, mientras que usar solo `G` o `(G + B) / 2` es sensible a halos verdosos/azulados en los bordes de la lente (`n0230`). La media `(G + R) / 2` exige brillo **simultáneo en rojo y verde** (tonalidad **amarillo-naranja brillante**), que corresponde precisamente a la firma de color del disco óptico y rechaza ambos tipos de falso positivo.

| Canal (`Gauss σ=50`, `ROI=600`) | ROIs 100 % disco (`dev`) | Error medio | Error mediano | Error máx. | Fallos |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `G` | 39/40 (97.5 %) | 80.4 px | 54.8 px | 1305.7 px | `n0230` (halo verdoso en borde) |
| `(G + B) / 2` | 39/40 (97.5 %) | 79.8 px | 52.0 px | 1351.5 px | `n0230` (halo verdoso-azulado) |
| **`(G + R) / 2`** | **40/40 (100.0 %)** | **47.6 px** | **51.9 px** | **77.0 px** | **Ninguno** |

### Otsu en dos etapas dentro de la ROI
1. **Primer intento (Otsu independiente en toda la ROI):**
   - Al aplicar Otsu directamente sobre toda la ROI en el canal `G`, la máscara de la copa captaba demasiada región exterior (parte del *rim*). Esto ocurre porque en la ROI hay tres clases (fondo de retina, *rim* y copa), y al dominar el fondo oscuro, Otsu colocaba el umbral entre el fondo y el disco entero en lugar de separar *rim* y copa.
2. **Solución adoptada (Otsu jerárquico en 2 etapas):**
   - **Etapa 1 — Disco (canal R en la ROI):** Otsu sobre los píxeles válidos de `roi_r` (`roi_r > 10`, excluyendo posibles bordes negros del exterior del ojo) seguido de cierre morfológico para obtener la máscara del disco.
   - **Etapa 2 — Copa (canal G dentro del disco):** Otsu restringido únicamente a los píxeles de `roi_g` que pertenecen al disco segmentado en la etapa 1, de modo que solo queden dos clases (*rim* vs. *copa*), seguido de cierre morfológico.

---

## 4. Situación actual: Transformada de Hough para elipses sobre bordes binarios

Incluso tras el cierre morfológico, las máscaras de Otsu conservan muescas provocadas por vasos gruesos y contornos irregulares que se alejan de la forma suave del *ground truth*. Además, el disco y la copa no siempre son círculos perfectos (por ejemplo, en `g0002` tienen una forma claramente elíptica e inclinada), por lo que un ajuste solo a círculos resulta insuficiente.

Para aproximar mejor la geometría real estamos probando:
1. **Extracción de bordes binarios:** se obtienen los bordes de 1 píxel directamente a partir de las máscaras binarias de Otsu tras el cierre (`mascara - erode(mascara)`), siguiendo la recomendación de trabajar sobre bordes binarios para simplificar la detección y evitar el ruido de textura y vasos de la imagen en escala de grises.
2. **Transformada de Hough para elipses (`houghElipse`):** se aplica el algoritmo de Xie & Ji (2002) sobre los bordes binarios del disco y de la copa (reduciendo la resolución de los puntos de borde para mantener tiempos bajos) para detectar la elipse `(centro, a, b, ángulo)` con mayor número de votos en cada máscara y generar una segmentación final más cercana al *ground truth*.

---

## 5. Seguimiento de pruebas y próximos pasos

- **Registro de ejecuciones:** cada ejecución de `main.py` guarda automáticamente en `ejecuciones/<imagen>_<metodo_tag>/` las cinco figuras generadas (canales y localización de ROI, máscaras/cierres Otsu, comparación Otsu vs. *ground truth*, bordes binarios con elipses de Hough, y comparación final de Hough frente al *ground truth*) junto con `params.txt`.
- **Pendiente de abordar:**
  - Evaluar el pipeline completo (Otsu en 2 etapas + Hough para elipses) sobre todo el conjunto de imágenes de desarrollo.
  - Implementar métricas cuantitativas frente al *ground truth* (error de localización del disco, Dice/IoU de disco y copa, y error en el cálculo del CDR).
  - Analizar los casos donde falle el ajuste (p. ej. si conviene filtrar la componente conexa principal antes de extraer bordes o ajustar restricciones de relación de ejes) y documentar comparativas para la memoria.

Importante: mirar si se puede ajustar algo más otsu
Explicar en memoria porque hacemos mascara - erode(mascara) y no canny
