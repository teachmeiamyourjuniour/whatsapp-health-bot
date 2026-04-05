import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skin_model_def import build_skin_model

IMG_SIZE      = (224, 224)
NUM_CLASSES   = 5
DATASET_DIR   = "C:\\Users\\Mukundraj\\OneDrive\\Desktop\\f\\skin_disease_model\\data\\skin_dataset_clean"
PHASE1_WEIGHTS = "skin_phase11_best.h5"
FINAL_WEIGHTS  = "skin_phase21_final.h5"

def build_finetune_model():
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
    base.trainable = False  # freeze first so weight loading matches exactly

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = models.Model(inputs, outputs)

    model.load_weights(PHASE1_WEIGHTS)
    print(f"Phase 1 weights loaded successfully.")

    # ── Unfreeze last 30 layers (more than 20 helps on skin datasets) ──────
# Unfreeze top 30 layers
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False

    # Keep ALL BatchNorm layers frozen regardless
    # (critical for small datasets — unfreezing BN destroys learned statistics)
    for layer in base.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False

    # Count from the FULL model, not just base, for an accurate picture
    total_trainable    = sum(1 for l in model.layers if l.trainable)
    base_trainable     = sum(1 for l in base.layers  if l.trainable)
    trainable_params   = sum(
        tf.size(w).numpy() for w in model.trainable_weights
    )

    print(f"Trainable layers  (full model) : {total_trainable}")
    print(f"Trainable layers  (base only)  : {base_trainable}")
    print(f"Trainable params              : {trainable_params:,}")
    return model


if __name__ == "__main__":
    # ── Same fixed datagen as Phase 1 ────────────────────────────────────
    datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.2,
        horizontal_flip=True,
        zoom_range=0.1,
    )
    train_gen = datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=16,
        class_mode="categorical", subset="training", seed=42
    )
    val_gen = datagen.flow_from_directory(
        DATASET_DIR, target_size=IMG_SIZE, batch_size=16,
        class_mode="categorical", subset="validation", seed=42, shuffle=False
    )

    model = build_finetune_model()

    # ── FIX 5: Cosine decay LR schedule instead of flat 1e-5 ─────────────
    # Starts at 5e-5 and decays smoothly — less risk of destroying Phase 1 gains.
    steps_per_epoch = len(train_gen)
    total_steps     = steps_per_epoch * 15
    lr_schedule     = tf.keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=5e-5, decay_steps=total_steps, alpha=1e-6
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr_schedule),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            FINAL_WEIGHTS, monitor="val_accuracy",
            save_best_only=True, save_weights_only=True, mode="max", verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=6, restore_best_weights=True, verbose=1
        ),
    ]

    print(f"\nPhase 2: Fine-tuning top 30 layers of EfficientNetB0...")
    model.fit(
        train_gen, validation_data=val_gen,
        epochs=15,
        callbacks=callbacks
    )
    print(f"\nDone. Final weights saved to '{FINAL_WEIGHTS}'")