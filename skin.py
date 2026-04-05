# ============================================================
#  Skin Disease Detection — IMPROVED Pipeline v2
#  Key upgrades:
#    1. EfficientNetB3 (stronger than MobileNetV2)
#    2. Class weights to fix imbalance
#    3. Better augmentation
#    4. Larger dense head
#    5. Longer fine-tuning
# ============================================================

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import seaborn as sns

# ─────────────────────────────────────────────
# STEP 1 — CONFIGURATION
# ─────────────────────────────────────────────
TRAIN_DIR   = "C:\\Users\\Mukundraj\\OneDrive\\Desktop\\f\\skin_disease_model\\data\\train"
TEST_DIR    = "C:\\Users\\Mukundraj\\OneDrive\\Desktop\\f\\skin_disease_model\\data\\test"

IMG_SIZE    = 300          # EfficientNetB3 native size
BATCH_SIZE  = 32
EPOCHS_P1   = 25           # Phase 1 epochs
EPOCHS_P2   = 15           # Phase 2 fine-tuning epochs
NUM_CLASSES = 23
MODEL_SAVE  = "skin_disease_v2_best.h5"


# ─────────────────────────────────────────────
# STEP 2 — DATA LOADING & AUGMENTATION
# ─────────────────────────────────────────────
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.15,
    brightness_range=[0.8, 1.2],   # vary brightness (skin lighting varies)
    channel_shift_range=20.0,      # vary colour tone
    fill_mode="nearest"
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False
)

class_names = list(train_generator.class_indices.keys())
print(f"\n✅ Found {len(class_names)} classes\n")


# ─────────────────────────────────────────────
# STEP 3 — COMPUTE CLASS WEIGHTS
# (fixes imbalance: Urticaria has 53, Acne has 312)
# ─────────────────────────────────────────────
labels = train_generator.classes
class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(labels),
    y=labels
)

# Use np.asscalar or float() on the underlying numpy value to be 100% sure
class_weight_dict = {int(i): float(class_weights_array[i]) for i in range(len(class_weights_array))}
print("⚖️  Class weights computed and scrubbed for JSON safety.\n")

# ─────────────────────────────────────────────
# STEP 4 — BUILD MODEL (EfficientNetB3)
# ─────────────────────────────────────────────
base_model = EfficientNetB3(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(256, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(NUM_CLASSES, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# ─────────────────────────────────────────────
# STEP 5 — CALLBACKS
# ─────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True, verbose=1),
    # IMPORTANT: We use save_weights_only=True to bypass the EagerTensor JSON error
    ModelCheckpoint(
        MODEL_SAVE, 
        monitor="val_accuracy", 
        save_best_only=True, 
        save_weights_only=True, # <--- THE MAGIC FIX
        verbose=1
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1)
]

# ─────────────────────────────────────────────
# STEP 6 — PHASE 1: Train top layers only
# ─────────────────────────────────────────────
print("\n🚀 Phase 1: Training new top layers...\n")

# Extra Safety: Clear the Keras session before starting
tf.keras.backend.clear_session()

history1 = model.fit(
    train_generator,
    epochs=EPOCHS_P1,
    validation_data=test_generator,
    class_weight=class_weight_dict,   
    callbacks=callbacks,
    verbose=1 # Ensure we see the progress bar
)

# ─────────────────────────────────────────────
# STEP 7 — PHASE 2: Fine-tune top 50 layers
# ─────────────────────────────────────────────
print("\n🔧 Phase 2: Fine-tuning top 50 layers of EfficientNetB3...\n")
base_model.trainable = True

for layer in base_model.layers[:-50]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

history2 = model.fit(
    train_generator,
    epochs=EPOCHS_P2,
    validation_data=test_generator,
    class_weight=class_weight_dict,
    callbacks=callbacks
)


# ─────────────────────────────────────────────
# STEP 8 — EVALUATE
# ─────────────────────────────────────────────
print("\n📊 Evaluating on test set...\n")
loss, accuracy = model.evaluate(test_generator)
print(f"\n✅ Test Accuracy : {accuracy * 100:.2f}%")
print(f"✅ Test Loss     : {loss:.4f}\n")

test_generator.reset()
y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

print("\n📋 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))


# ─────────────────────────────────────────────
# STEP 9 — PLOT RESULTS
# ─────────────────────────────────────────────
def plot_history(h1, h2=None):
    acc   = h1.history["accuracy"]
    val   = h1.history["val_accuracy"]
    loss  = h1.history["loss"]
    vloss = h1.history["val_loss"]

    if h2:
        acc   += h2.history["accuracy"]
        val   += h2.history["val_accuracy"]
        loss  += h2.history["loss"]
        vloss += h2.history["val_loss"]

    e = range(len(acc))
    plt.figure(figsize=(14, 5))
    plt.subplot(1, 2, 1)
    plt.plot(e, acc,  label="Train Accuracy")
    plt.plot(e, val,  label="Val Accuracy")
    plt.axvline(x=len(h1.history["accuracy"]) - 1, color="gray",
                linestyle="--", label="Fine-tune start")
    plt.title("Accuracy over Epochs"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(e, loss,  label="Train Loss")
    plt.plot(e, vloss, label="Val Loss")
    plt.axvline(x=len(h1.history["loss"]) - 1, color="gray",
                linestyle="--", label="Fine-tune start")
    plt.title("Loss over Epochs"); plt.legend()
    plt.tight_layout()
    plt.savefig("training_history_v2.png", dpi=150)
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(18, 16))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix — v2")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("confusion_matrix_v2.png", dpi=150)
    plt.show()


plot_history(history1, history2)
plot_confusion_matrix(y_true, y_pred, class_names)


# ─────────────────────────────────────────────
# STEP 10 — PREDICT ON A SINGLE IMAGE
# ─────────────────────────────────────────────
def predict_image(img_path, model, class_names, top_k=3):
    img = tf.keras.utils.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)[0]
    top_indices = np.argsort(preds)[::-1][:top_k]

    print(f"\n🔍 Image: {img_path}")
    print(f"🩺 Top {top_k} Predictions:")
    for i, idx in enumerate(top_indices):
        print(f"   {i+1}. {class_names[idx]:<60} {preds[idx]*100:.1f}%")

# Example:
# predict_image("data/test/Eczema Photos/some_image.jpg", model, class_names)