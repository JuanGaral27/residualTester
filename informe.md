# ♻️ Clasificador de Residuos mediante Visión Artificial

Este proyecto utiliza **Deep Learning** para identificar y clasificar automáticamente diferentes tipos de residuos sólidos. El objetivo es proporcionar una herramienta tecnológica que facilite la separación de basura en origen, optimizando los procesos de reciclaje.

## 🧠 Descripción Técnica del Modelo

Para este proyecto, se implementó una técnica llamada **Transfer Learning** (Aprendizaje por Transferencia). En lugar de entrenar una red desde cero, utilizamos **MobileNetV2**, una arquitectura desarrollada por Google que ya conoce cómo identificar formas y texturas básicas, y la adaptamos a nuestro dataset específico de residuos.



### Componentes de la Red:
* **Base:** MobileNetV2 (Pre-entrenada en ImageNet).
* **Capas de Preprocesamiento:** Integradas en el modelo para realizar normalización y aumento de datos (Data Augmentation) en tiempo real.
* **Cuello de Botella:** `GlobalAveragePooling2D` para reducir la complejidad sin perder información clave.
* **Capa de Salida:** Densa con activación **Softmax**, que entrega la probabilidad para cada categoría de residuo.

---

## 📂 Estructura del Sistema

El software está dividido en tres módulos principales para garantizar escalabilidad y orden:

1.  **`train.py` (Entrenamiento):** Gestiona la carga de imágenes, el aumento de datos y las dos fases de entrenamiento (congelado y ajuste fino o *fine-tuning*).
2.  **`evaluate.py` (Métricas):** Analiza el rendimiento del modelo utilizando datos que la IA nunca ha visto, generando una matriz de confusión y un reporte de precisión/sensibilidad.
3.  **`app.py` (Interfaz):** Una aplicación web interactiva desarrollada en **Streamlit** que permite a un usuario final subir fotos y obtener resultados inmediatos.



---

## 🛠️ Tecnologías Utilizadas

* **Python 3.12**
* **TensorFlow / Keras:** Motor de la Inteligencia Artificial.
* **Streamlit:** Framework para la interfaz de usuario web.
* **Scikit-Learn:** Herramientas para el análisis de métricas.
* **Pandas & Numpy:** Manejo eficiente de matrices y datos.
* **Matplotlib & Seaborn:** Visualización de resultados.

---

## 📊 Mejoras Implementadas

Durante el desarrollo, se corrigieron problemas críticos detectados en versiones anteriores:

* **Rutas Dinámicas:** Se implementó un sistema de rutas absolutas basadas en el archivo actual (`os.path.abspath`), eliminando errores de "Archivo no encontrado" en diferentes sistemas operativos.
* **Optimización oneDNN:** Configuración adaptada para aprovechar las instrucciones de hardware de los procesadores modernos.
* **Robustez en Evaluación:** El evaluador ahora detecta automáticamente las clases presentes en la carpeta de prueba, evitando errores de desajuste de etiquetas.

---

## 🚀 Cómo Ejecutar el Proyecto

### Entrenamiento
```bash
python src/train.py

python src/evaluate.py

streamlit run src/app.py