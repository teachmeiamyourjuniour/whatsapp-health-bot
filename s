# debug_skin.py
import tensorflow as tf
from tensorflow.keras import layers, models
import h5py

tf.keras.backend.clear_session()

# Build skeleton exactly as in server
base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3)
)
base.trainable = False
inputs = tf.keras.Input(shape=(224, 224, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.BatchNormalization()(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.4)(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(5, activation="softmax")(x)
skin_model = models.Model(inputs, outputs)

# Print live model layer names
print("\n--- LIVE MODEL LAYERS ---")
for layer in skin_model.layers:
    print(f"  {layer.name}: {[w.shape for w in layer.weights]}")

# Print h5 file layer names
print("\n--- H5 FILE LAYERS ---")
with h5py.File("skin_phase21_final.h5", "r") as f:
    print("Keys:", list(f.keys()))
    