import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from tensorflow.keras import regularizers
from skin_model_def import build_skin_model


IMG_SIZE    = (224, 224)
NUM_CLASSES = 5
DATASET_DIR = "C:\\Users\\Mukundraj\\OneDrive\\Desktop\\f\\skin_disease_model\\data\\skin_dataset_clean"
MODEL_SAVE  = "skin_phase11_best.h5"

def build_model():
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)   # <-- added
    x = layers.Dropout(0.5)(x)                                       # increased from 0.4
    x = layers.Dense(128, activation="relu",
                     kernel_regularizer=regularizers.l2(1e-4))(x)   # <-- added
    x = layers.Dropout(0.4)(x)                                       # increased from 0.3
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return models.Model(inputs, outputs), base


if __name__ == "__main__":
 # phase1_train.py — add these fixes

# ── FIX 1: Much heavier augmentation ─────────────────────────────────────
# Your dataset is tiny. Augmentation IS your extra data.

    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        # Geometric
        horizontal_flip=True,
        vertical_flip=True,           # skin lesions have no fixed orientation
        rotation_range=360,           # full rotation, lesions can appear at any angle
        zoom_range=0.2,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.1,
        # Color/intensity (critical for skin images)
        brightness_range=[0.75, 1.25],
        channel_shift_range=20.0,     # simulates different skin tones and lighting
        fill_mode="reflect",          # better than 'nearest' for skin textures
    )

    # ── FIX 2: ONE datagen object, seed-locked splits ─────────────────────
    # Both subsets must come from the SAME generator with the SAME seed,
    # otherwise they reshuffle independently and you get data leakage.
    train_gen = datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=32,
        class_mode="categorical", subset="training", seed=42
    )
    val_gen = datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=32,
        class_mode="categorical", subset="validation", seed=42, shuffle=False
    )

# ── FIX 3: Handle class imbalance (CORRECTED) ─────────────────────────
from collections import Counter
    
    # Get the counts of each class
class_counts = Counter(train_gen.classes)
total_samples = sum(class_counts.values())
    
    # Calculate weights: (Total Samples) / (Number of Classes * Count of that class)
class_weight = {
        class_id: total_samples / (NUM_CLASSES * count)
        for class_id, count in class_counts.items()
    }
print("Class weights:", class_weight)

model, base = build_model()
model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        "skin_phase11_best.h5",        # ← explicit string, first positional arg
        monitor="val_accuracy",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
        verbose=1
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=7,
        restore_best_weights=True, verbose=1
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss", factor=0.5,
        patience=3, min_lr=1e-6, verbose=1
    ),
]

# ── FIX 4: Use MixUp augmentation (biggest accuracy booster for tiny datasets)
# Add this AFTER the datagen setup, as a custom generator wrapper.
import numpy as np

def mixup_generator(gen, alpha=0.2):
    """Blends pairs of images and their labels — forces the model to
    learn smooth decision boundaries instead of memorizing samples."""
    while True:
        X1, y1 = next(gen)
        X2, y2 = next(gen)
        lam = np.random.beta(alpha, alpha)
        X = lam * X1 + (1 - lam) * X2
        y = lam * y1 + (1 - lam) * y2
        yield X, y

# Then replace:
# model.fit(train_gen, ...)
# With:
# model.fit(mixup_generator(train_gen), steps_per_epoch=len(train_gen), ...)

print(f"\nPhase 1: Training classification head only...")
model.fit(
        train_gen, validation_data=val_gen,
        epochs=20,                          # early stopping will cut this short
        class_weight=class_weight,
        callbacks=callbacks
    )
print(f"Phase 1 complete. Best weights saved to '{MODEL_SAVE}'")