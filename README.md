# ♻️ Sistema de Clasificación de Residuos con Inteligencia Artificial

Este proyecto implementa una solución de **Visión Artificial** basada en Deep Learning para la identificación y clasificación automática de residuos domésticos, facilitando el proceso de reciclaje.

## 🚀 Características Principales
- **Arquitectura:** Transfer Learning utilizando **MobileNetV2** (pre-entrenada en ImageNet).
- **Dataset:** Clasificación multiclase (Plástico, Cartón, Vidrio, Metal, Papel, etc.).
- **Tecnología:** TensorFlow 2.x, Keras, Streamlit y OpenCV.
- **Optimización:** Pipeline de datos avanzado con `tf.data` y capas de preprocesamiento integradas.

---

## Arquitectura del Modelo

El modelo utiliza una red neuronal convolucional (CNN) optimizada para dispositivos móviles y eficiencia computacional.

1.  **Capa de Entrada:** Imágenes de 224x224x3 píxeles.
2.  **Aumento de Datos (Augmentation):** Capas integradas de rotación, zoom y volteo horizontal para mejorar la generalización.
3.  **Base (Backbone):** MobileNetV2 (congelada inicialmente, luego ajustada con *Fine-Tuning*).
4.  **Cabecera Global:** - `GlobalAveragePooling2D` para reducir dimensiones.
    - `Dropout (0.3)` para evitar el sobreajuste (overfitting).
    - `Dense` con activación **Softmax** para la clasificación final.

---

## 📂 Estructura del Proyecto

```text
proyecto1/
├── data/               # Datasets divididos en train, val y test
├── models/             # Modelos exportados (.h5) y etiquetas (.json)
├── reports/            # Gráficos de rendimiento y matrices de confusión
└── src/                # Código fuente
    ├── app.py          # Interfaz web con Streamlit
    ├── train.py        # Script de entrenamiento avanzado
    └── evaluate.py     # Script de métricas y evaluación
