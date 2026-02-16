import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np
import os

# ==========================================
# 1. RUTA Y VERIFICACIÓN DE DATOS (Paso 2)
# ==========================================
PATH_DATASET = r'C:\Users\jdgar\Documents\Documentos\IA\proyecto1\dataset'

print("--- REVISANDO CARPETAS ---")
if not os.path.exists(PATH_DATASET):
    print(f"ERROR: La ruta {PATH_DATASET} no existe.")
    exit()

# Este bloque busca CUALQUIER archivo para decirte qué hay dentro
formatos_validos = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.jfif')
conteo_total = 0

for root, dirs, files in os.walk(PATH_DATASET):
    categoria = os.path.basename(root)
    imagenes_en_carpeta = [f for f in files if f.lower().endswith(formatos_validos)]
    if categoria != "dataset":
        print(f"Carpeta '{categoria}': {len(imagenes_en_carpeta)} imágenes encontradas.")
        conteo_total += len(imagenes_en_carpeta)

if conteo_total == 0:
    print("FATAL: No se encontraron imágenes. Revisa que las fotos no sean .webp o que no estén en carpetas vacías.")
    exit()

# ==========================================
# 2. PREPROCESAMIENTO (Paso 3)
# ==========================================
IMG_SIZE = (224, 224)
BATCH_SIZE = 8 # Reducido por si tienes pocas fotos

print("\n--- CARGANDO DATASET ---")
# Entrenamiento
train_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_DATASET,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

# Validación
val_ds = tf.keras.utils.image_dataset_from_directory(
    PATH_DATASET,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical'
)

class_names = train_ds.class_names
print(f"Clases listas para entrenar: {class_names}")

# ==========================================
# 3. MODELO CNN - TRANSFER LEARNING (Paso 4)
# ==========================================
# Usamos MobileNetV2 (Eficiente para computadoras estándar)
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False 

model = models.Sequential([
    layers.Rescaling(1./255), # Normalización exigida
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(len(class_names), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# ==========================================
# 4. ENTRENAMIENTO Y GUARDADO
# ==========================================
print("\n--- INICIANDO ENTRENAMIENTO ---")
history = model.fit(train_ds, validation_data=val_ds, epochs=10)

# Guardar el "cerebro" para la interfaz
model.save('clasificador_residuos.h5')
print("\n¡LISTO! Modelo guardado como 'clasificador_residuos.h5'")

# Graficar resultados (Paso 6)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del Modelo')
plt.legend()
plt.show()