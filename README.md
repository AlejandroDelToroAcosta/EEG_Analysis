# Análisis de señales EEG con redes neuronales para predecir estados de divagación mental

Trabajo de Fin de Grado · Grado en Ciencia e Ingeniería de Datos · ULPGC
Escuela de Ingeniería Informática · Curso 2025-2026

---

## Descripción del proyecto

Ante el repunte de la ansiedad y el elevado consumo de fármacos ansiolíticos, la meditación se posiciona como una alternativa clínica eficaz y respaldada por organismos como la OMS. Sin embargo, su práctica se ve dificultada por la **divagación mental (mind wandering)**, un fenómeno que hasta ahora se evaluaba mediante autoinformes subjetivos.

Este proyecto propone el análisis de señales de electroencefalografía (EEG) mediante técnicas de Machine Learning y Deep Learning para predecir de forma automática y objetiva la transición entre un estado de atención focalizada (**On-Task**) y un estado de divagación mental (**Mind Wandering**), sentando las bases para futuros sistemas de neurofeedback en tiempo real.

El desarrollo sigue la metodología **CRISP-DM**.

---

## Conjunto de datos

Se utiliza el dataset público de meditación de **Brandmeyer & Delorme (2017)**, resultado de un experimento en el que 24 sujetos se someten a sesiones de meditación de entre 45 y 90 minutos, respondiendo periódicamente a una batería de preguntas sobre su profundidad meditativa, nivel de divagación y somnolencia.

| Elemento | Valor |
|---|---|
| Sujetos evaluados | 24 |
| Canales EEG activos | 80 (64 Biosemi + 16 auxiliares) |
| Frecuencia de muestreo | 256 Hz (submuestreada desde 2048 Hz) |
| Sesiones | 2 por sujeto |
| Duración de las épocas | 10 s previos a cada interrupción (Q1/Q2/Q3) |

El etiquetado binario (`On-Task` = 0, `Mind Wandering` = 1) se deriva de las respuestas de autoinforme registradas mediante teclado táctil durante la meditación.

---

## Fases del proyecto

### 1. Preprocesamiento de señales EEG

Pipeline implementado en **MNE-Python** compuesto por cinco etapas secuenciales:

1. **Filtrado y segmentación**: filtro band-pass [0.1, 42] Hz y segmentación en épocas de 10 s.
2. **Inspección visual**: detección y descarte manual de canales/épocas con ruido evidente.
3. **ICA (Análisis de Componentes Independientes)**: descomposición de los 80 canales originales en 14 componentes independientes para aislar artefactos oculares y musculares.
4. **Inspección de componentes ICA**: verificación visual de las componentes resultantes.
5. **Reconstrucción y control final**: eliminación de componentes ruidosas, re-corrección de línea base y descarte definitivo de épocas remanentes.

### 2. Extracción de características (enfoque clásico — SVM)

A partir de la señal limpia se extraen dos familias de características mediante análisis tiempo-frecuencia (filtrado FIR + Transformada de Hilbert) en las bandas **Alpha (8-12 Hz)** y **Theta (4-8 Hz)**:

- **Potencia local instantánea** por canal y banda.
- **Conectividad funcional (ISPC / Intersite Phase Clustering)**: sincronía de fase entre pares de electrodos, empleada como proxy de la comunicación inter-regional asociada a la Red Neuronal por Defecto (DMN).

### 3. Representación espectrotemporal (enfoque profundo — CNN)

Para la red convolucional, la señal se transforma mediante la **Transformada Wavelet Continua (CWT)** con wavelet de Morlet compleja, generando escalogramas de dimensión `[Canales × Frecuencias × Tiempo]` que preservan la topología espacio-temporal de la señal, evitando el aplanado manual de características.

### 4. Modelos de clasificación

| Modelo | Entrada | Enfoque de validación |
|---|---|---|
| **SVM** (Máquina de Vectores de Soporte) | Vector plano de potencia + ISPC | Intersujeto (LOGO-CV) e Intrasujeto (LOOCV) |
| **EEG-NeXt** (CNN, backbone ConvNeXt) | Escalogramas CWT (5 canales parieto-occipitales) | Intersujeto (5-fold, cross-subject) |

`EEG-NeXt` incorpora alineación en el espacio Euclídeo (normalización de la covarianza por sujeto) como mecanismo de adaptación de dominio, bloques residuales con convoluciones *depthwise*, LayerNorm, activación GELU y *data augmentation* espectrotemporal (ruido gaussiano, *frequency masking* y *time masking*).

### 5. Validación estadística

Todos los resultados relevantes se contrastan mediante **tests de permutaciones no paramétricos** (barajado de etiquetas) para descartar que el rendimiento se deba a memorización o correlaciones espurias, y no a la extracción de patrones biológicos reales.

---

## Resultados principales

| Configuración | Enfoque | Accuracy | AUC |
|---|---|---|---|
| SVM — Potencia local | Intersujeto | 0.53 | 0.51 |
| SVM — ISPC | Intersujeto | 0.57 | 0.45 |
| SVM — Híbrido (Potencia + ISPC) | Intersujeto | 0.57 | 0.54 |
| SVM — Híbrido, corregido (sin data leakage) | **Intrasujeto** | **≈0.65** (media) | — |
| **EEG-NeXt** (base) | Intersujeto (LOGO-CV) | **0.62 ± 0.01** | 0.61 ± 0.04 |
| **EEG-NeXt** + Data Augmentation | Intersujeto (LOGO-CV) | 0.61 ± 0.05 | **0.63 ± 0.04** |

**Principales hallazgos:**

- El enfoque **SVM intersujeto** fracasa sistemáticamente (rendimiento cercano al azar), confirmando la fuerte variabilidad interindividual de la señal EEG (*covariate shift*).
- El **SVM intrasujeto**, tras corregir un problema de *data leakage* detectado en un experimento preliminar, replica los rangos de precisión reportados en la literatura de referencia (Jin et al., 2019).
- La red **EEG-NeXt** logra generalizar entre sujetos no vistos, superando de forma robusta y estadísticamente significativa (p < 0.05 en el test de permutaciones) el umbral del azar, sin necesidad de calibración individual — constituyendo la principal aportación de este trabajo frente al estado del arte.
- Las técnicas de *Data Augmentation* actúan como regularizador, mejorando la estabilidad del AUC a costa de una pequeña variación en el Accuracy.

---

## Stack tecnológico

| Área | Tecnologías |
|---|---|
| Lenguaje | Python 3.12 |
| Procesamiento de señales EEG | MNE-Python 1.10.2 |
| Deep Learning | PyTorch 2.2.1 |
| Machine Learning clásico | scikit-learn 1.8.0 |
| Manejo de datos | numpy, pandas |
| Visualización | matplotlib, seaborn |
| Cómputo | GPU T4 (Google Colab) para entrenamiento de EEG-NeXt |

---

## Limitaciones y trabajo futuro

- **Tamaño muestral reducido por sujeto** (< 20 épocas en la mayoría de casos), lo que limita la potencia estadística de los tests de permutaciones a nivel individual.
- **Desbalanceo de clases** nativo entre estados On-Task y Mind Wandering.
- Líneas futuras: ampliación del dataset, inspección de artefactos asistida por expertos clínicos, exploración de arquitecturas basadas en atención (Vision Transformers para biseñales) y despliegue en dispositivos EEG portátiles de bajo número de canales para monitorización cognitiva en tiempo real.

---


## Autor

Alejandro Lorenzo Del Toro Acosta
Grado en Ciencia e Ingeniería de Datos · ULPGC
Tutorizado por Javier Sánchez Medina
Curso 2025-2026
