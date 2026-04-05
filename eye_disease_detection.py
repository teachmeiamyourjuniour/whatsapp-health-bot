# ============================================================
# Eye Disease Detection — EfficientNetB0 (Ocular-Net)
# Finalized for B.Tech Demo — Phase 1 Stability Only
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import tensorflow.keras.backend as K

# ─────────────────────────────────────────────
# 1. GLOBAL SETTINGS (Required for Import)
# ─────────────────────────────────────────────
IMG_SIZE = (224, 224)

def focal_loss(gamma=2.0):
    def loss_fn(y_true, y_pred):
        y_pred = K.clip(y_pred, 1e-7, 1.0)
        cross_entropy = -y_true * K.log(y_pred)
        weight = K.pow(1.0 - y_pred, gamma) * y_true
        return K.sum(weight * cross_entropy, axis=-1)
    return loss_fn

def build_model(num_classes=3):
    """Builds the architecture skeleton."""
    base_model = EfficientNetB0(
        include_top=False, 
        weights="imagenet", 
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )
    base_model.trainable = False  # Keep the base frozen for 61% stability

    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = models.Model(inputs, outputs)
    return model, base_model

# ─────────────────────────────────────────────
# 2. TRAINING LOCK (Only runs if you execute this script)
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("🚀 Starting the Final Winning Run (Phase 1 Only)...")
    
    # Paths
    DATASET_DIR = "C:\\Users\\Mukundraj\\OneDrive\\Desktop\\f\\dataset"
    
    # Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255, 
        validation_split=0.3, 
        horizontal_flip=True,
        zoom_range=0.15
    )
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.3)

    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=32, 
        class_mode="categorical", subset="training", seed=42
    )
    val_gen = val_datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=32, 
        class_mode="categorical", subset="validation", shuffle=False, seed=42
    )

    # Initialize
    model, base_model = build_model()

    # Compile with the 61% Winning Settings
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), 
        loss=focal_loss(gamma=1.0), 
        metrics=["accuracy"]
    )

    # CHECKPOINT: Automatically saves the BEST version of the model to disk
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        "eye_model_60_FINAL_weights.h5", 
        monitor='val_accuracy', 
        save_best_only=True, 
        save_weights_only=True, 
        mode='max', 
        verbose=1
    )

    # Execute Training
    print("\n🔥 Training custom layers for 20 epochs...")
    model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=20, 
        callbacks=[checkpoint]
    )

    print("\n✅ WINNER SAVED! Your best weights are in 'eye_model_60_FINAL_weights.h5'.")
    print("You can now close this and run your whatsapp_server.py.")