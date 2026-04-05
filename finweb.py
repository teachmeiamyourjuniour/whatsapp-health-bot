import gradio as gr
import cv2
import numpy as np
import os
import h5py
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications.efficientnet import preprocess_input as skin_preprocess
from medical_bot import chatbot_response
from explainability import generate_and_save_heatmap
from preprocessor import extract_eye_crop
from eye_disease_detection import build_model as build_eye_model

tf.keras.backend.clear_session()

eye_model, _ = build_eye_model(num_classes=3)
eye_model.load_weights('eye_model_60_FINAL_weights.h5', by_name=True)
print("✅ Eye model loaded")

def load_skin_model():
    base = tf.keras.applications.EfficientNetB0(include_top=False, weights="imagenet", input_shape=(224, 224, 3))
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
    model = models.Model(inputs, outputs)
    with h5py.File("skin_phase21_final.h5", "r") as f:
        bn_name = [l.name for l in model.layers if "batch_normalization" in l.name][0]
        dense_names = sorted([l.name for l in model.layers if l.name.startswith("dense")],
                             key=lambda x: int(x.split("_")[1]) if "_" in x else 0)
        bn = f["batch_normalization"]["batch_normalization"]
        model.get_layer(bn_name).set_weights([bn["gamma:0"][()], bn["beta:0"][()],
                                              bn["moving_mean:0"][()], bn["moving_variance:0"][()]])
        for live_name, h5_key in zip(dense_names, ["dense", "dense_1", "dense_2"]):
            g = f[h5_key][h5_key]
            model.get_layer(live_name).set_weights([g["kernel:0"][()], g["bias:0"][()]])
    return model

skin_model = load_skin_model()
print("✅ Skin model loaded")

EYE_CLASSES = ['Cataract', 'Diabetic Retinopathy', 'Normal']
SKIN_CLASSES = ['Acne', 'Melanoma', 'Scaly_Lesions', 'Vitiligo_Pigmentation', 'Warts_Viral']

def analyze_body_ailment(img, mode):
    if img is None:
        return None, "⚠️ Please upload an image.", "No advice available."
    temp_path = f"web_{mode.lower()}_upload.jpg"
    cv2.imwrite(temp_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    try:
        if mode == "Ocular (Eye)":
            processed_path = extract_eye_crop(temp_path)
            img_to_pred = cv2.imread(processed_path)
            img_resized = cv2.resize(img_to_pred, (224, 224)) / 255.0
            preds = eye_model.predict(np.expand_dims(img_resized, axis=0))[0]
            classes = EYE_CLASSES
            current_model = eye_model
            final_path = processed_path
        else:
            img_resized = cv2.resize(img, (224, 224))
            img_array = np.expand_dims(img_resized, axis=0)
            img_preprocessed = skin_preprocess(img_array.astype('float32'))
            preds = skin_model.predict(img_preprocessed)[0]
            classes = SKIN_CLASSES
            current_model = skin_model
            final_path = temp_path

        idx = np.argmax(preds)
        diagnosis = classes[idx]
        confidence = preds[idx] * 100
        heatmap_out = f"static/web_heatmap_{mode.lower()}.jpg"
        heatmap_path = generate_and_save_heatmap(final_path, current_model, output_path=heatmap_out)
        heatmap_img = cv2.cvtColor(cv2.imread(heatmap_path), cv2.COLOR_BGR2RGB)
        advice = chatbot_response(f"The patient has {diagnosis}. Explain it and give 2 precautions under 100 words.", target_lang='en')
        result_text = f"## AI Result: {diagnosis} ({confidence:.1f}%)"
        return heatmap_img, result_text, advice
    except Exception as e:
        return None, f"❌ Analysis Error: {str(e)}", "Please check system logs."

with gr.Blocks(theme=gr.themes.Soft(), title="Methodist Multi-Disease AI") as demo:
    gr.Markdown("# 🏥 Multimodal AI Triage & Diagnostic Dashboard")
    gr.Markdown("### Final Year Project - Methodist College of Engineering & Technology")
    with gr.Tabs():
        with gr.TabItem("👁️ Ocular Analysis"):
            with gr.Row():
                with gr.Column():
                    eye_input = gr.Image(label="Upload Eye Image")
                    eye_btn = gr.Button("Analyze Eye", variant="primary")
                with gr.Column():
                    eye_heatmap = gr.Image(label="AI Vision (Grad-CAM)")
                    eye_result = gr.Markdown()
                    eye_advice = gr.Textbox(label="Medical Guidance", lines=4)
            eye_btn.click(fn=lambda img: analyze_body_ailment(img, "Ocular (Eye)"),
                          inputs=eye_input, outputs=[eye_heatmap, eye_result, eye_advice])
        with gr.TabItem("🩹 External Body Ailments (Skin)"):
            with gr.Row():
                with gr.Column():
                    skin_input = gr.Image(label="Upload Lesion Image")
                    skin_btn = gr.Button("Analyze Skin", variant="primary")
                with gr.Column():
                    skin_heatmap = gr.Image(label="AI Vision (Grad-CAM)")
                    skin_result = gr.Markdown()
                    skin_advice = gr.Textbox(label="Medical Guidance", lines=4)
            skin_btn.click(fn=lambda img: analyze_body_ailment(img, "External Body"),
                           inputs=skin_input, outputs=[skin_heatmap, skin_result, skin_advice])
    gr.HTML("<hr>")
    gr.Markdown("### 💬 Medical Knowledge Base Chat (Multilingual RAG)")
    gr.ChatInterface(fn=lambda msg, history: chatbot_response(msg, target_lang='en'))

if __name__ == "__main__":
    if not os.path.exists("static"): os.makedirs("static")
    demo.launch(inbrowser=True)