import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess

# Build combined grad model ONCE at import time — not inside the function
_grad_models = {}

def get_grad_model(model):
    model_id = id(model)
    if model_id not in _grad_models:
        base_model = model.get_layer('efficientnetb0')
        conv_output = base_model.get_layer('top_activation').output

        # Walk UP from conv_output through the rest of model's layers
        # to get final predictions — all in ONE connected graph
        x = conv_output
        # Get all layers after efficientnetb0 in the full model
        found = False
        for layer in model.layers:
            if layer.name == 'efficientnetb0':
                found = True
                continue
            if found:
                x = layer(x)

        _grad_models[model_id] = tf.keras.models.Model(
            inputs=base_model.input,
            outputs=[conv_output, x]   # conv map + final predictions
        )
        print(f"✅ Grad model built for {model.name}")
    return _grad_models[model_id]


def generate_and_save_heatmap(img_path, model, output_path="static/heatmap_result.jpg", mode="EYE"):
    try:
        img = cv2.imread(img_path)
        if img is None:
            print("❌ Could not read image")
            return None

        img_res = cv2.resize(img, (224, 224))

        if mode == "SKIN":
            img_array = np.expand_dims(img_res.astype('float32'), axis=0)
            img_array = efficientnet_preprocess(img_array)
        else:
            img_array = np.expand_dims(img_res.astype('float32') / 255.0, axis=0)

        img_tensor = tf.cast(img_array, tf.float32)

        # Get single connected grad model
        grad_model = get_grad_model(model)

        with tf.GradientTape() as tape:
            tape.watch(img_tensor)
            last_conv_output, preds = grad_model(img_tensor, training=False)
            top_class_index = int(tf.argmax(preds[0]))
            class_channel = preds[:, top_class_index]

        grads = tape.gradient(class_channel, last_conv_output)

        if grads is None:
            print("❌ Gradient still None")
            cv2.imwrite(output_path, img_res)
            return output_path

        pooled_grads = tf.reduce_mean(grads, axis=[0, 1, 2]).numpy()
        conv_np = last_conv_output[0].numpy()

        heatmap = np.zeros(conv_np.shape[:2], dtype=np.float32)
        for i, w in enumerate(pooled_grads):
            heatmap += w * conv_np[:, :, i]

        heatmap = np.maximum(heatmap, 0)
        heatmap = heatmap / (np.max(heatmap) + 1e-10)

        heatmap_rescaled = cv2.resize(heatmap, (224, 224))
        heatmap_rescaled = np.uint8(255 * heatmap_rescaled)
        heatmap_color = cv2.applyColorMap(heatmap_rescaled, cv2.COLORMAP_JET)
        superimposed = cv2.addWeighted(img_res, 0.4, heatmap_color, 0.6, 0)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, superimposed)
        print(f"✅ Heatmap saved: {output_path}")
        return output_path

    except Exception as e:
        print(f"❌ Heatmap Error: {e}")
        return None