import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# 1. LOAD MODEL
model = load_model('eye_disease_model.h5')

# 2. AUTO-FIND AN IMAGE
folder_path = 'dataset/cataract'
all_images = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

if not all_images:
    print(f"❌ Error: No images found in {folder_path}")
    exit()

img_name = all_images[0] 
img_path = os.path.join(folder_path, img_name)
print(f"✅ Generating Heatmap for: {img_path}")

def get_gradcam_heatmap(model, img_array):
    # 1. Get the top prediction FIRST, outside the tape
    preds = model.predict(img_array)
    top_class_index = np.argmax(preds[0])

    # 2. Setup the internal model to look at 'Conv_1'
    base_model = model.get_layer('mobilenetv2_1.00_224')
    grad_model = tf.keras.models.Model(
        [base_model.inputs], 
        [base_model.get_layer('Conv_1').output, base_model.output]
    )

    # 3. Use the Tape ONLY for the gradients
    with tf.GradientTape() as tape:
        last_conv_layer_output, base_preds = grad_model(img_array)
        class_channel = base_preds[:, top_class_index]

    # 4. Calculate gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    
    # 5. Normalize
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()

# 4. PREPROCESS THE IMAGE (Defining img_array HERE)
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 5. RUN HEATMAP GENERATION
heatmap = get_gradcam_heatmap(model, img_array)

# 6. MERGE AND SAVE
img_cv2 = cv2.imread(img_path)
img_cv2 = cv2.resize(img_cv2, (224, 224))
heatmap_rescaled = cv2.resize(heatmap, (224, 224))
heatmap_rescaled = np.uint8(255 * heatmap_rescaled)
heatmap_color = cv2.applyColorMap(heatmap_rescaled, cv2.COLORMAP_JET)

superimposed_img = heatmap_color * 0.4 + img_cv2
output_name = 'eye_analysis_heatmap.jpg'
cv2.imwrite(output_name, superimposed_img)

print(f"🔥 Success! Heatmap saved as '{output_name}'")

# Display the result
plt.imshow(cv2.cvtColor(superimposed_img.astype('uint8'), cv2.COLOR_BGR2RGB))
plt.title(f"XAI Diagnosis: {img_name}")
plt.axis('off')
plt.show()