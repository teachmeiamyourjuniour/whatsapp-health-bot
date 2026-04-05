# convert_weights.py — final working version
import tensorflow as tf
from tensorflow.keras import layers, models
import h5py
import numpy as np

tf.keras.backend.clear_session()

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

with h5py.File("skin_phase21_final.h5", "r") as f:
    
    # BatchNormalization — 4 weights: gamma, beta, moving_mean, moving_variance
    bn = f["batch_normalization"]["batch_normalization"]
    skin_model.get_layer("batch_normalization").set_weights([
        bn["gamma:0"][()],
        bn["beta:0"][()],
        bn["moving_mean:0"][()],
        bn["moving_variance:0"][()]
    ])
    print("✅ Loaded: batch_normalization")

    # Dense layers — 2 weights each: kernel, bias (order matters)
    for layer_name in ["dense", "dense_1", "dense_2"]:
        g = f[layer_name][layer_name]
        skin_model.get_layer(layer_name).set_weights([
            g["kernel:0"][()],
            g["bias:0"][()]
        ])
        print(f"✅ Loaded: {layer_name}")

skin_model.save("skin_final_full.h5")
print("✅ Saved as skin_final_full.h5")