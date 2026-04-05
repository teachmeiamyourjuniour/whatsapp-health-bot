import os
import requests
import cv2
import numpy as np
import threading
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from twilio.rest import Client
import tensorflow as tf
import h5py 
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.applications.efficientnet import preprocess_input as skin_preprocess
from tensorflow.keras import regularizers 


from eye_disease_detection import build_model as build_eye_model
from medical_bot import chatbot_response
from explainability import generate_and_save_heatmap
from preprocessor import extract_eye_crop
from skin_model_def import build_skin_model
import traceback
skin_model, _ = build_skin_model()

print("--- INITIALIZING MULTI-DISEASE HEALTHCARE BOT ---")
tf.keras.backend.clear_session()

app = Flask(__name__)
user_modes = {} 
user_languages = {}


eye_model, eye_base = build_eye_model(num_classes=3)
try:
    eye_model.load_weights("eye_model_60_FINAL_weights.h5", by_name=True)
    print("✅ Ocular-Net Online!")
except Exception as e:
    print(f"❌ Eye Weight Error: {e}")


def load_skin_model():
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
    model = models.Model(inputs, outputs)

  
    bn_name    = [l.name for l in model.layers if "batch_normalization" in l.name][0]
    dense_names = [l.name for l in model.layers if l.name.startswith("dense")]
    # dense_names will be e.g. ['dense_3', 'dense_4', 'dense_5'] — sorted by index
    dense_names.sort(key=lambda x: int(x.split("_")[1]) if "_" in x else 0)

    print(f"Layer names — BN: {bn_name}, Dense: {dense_names}")

    with h5py.File("skin_phase21_final.h5", "r") as f:
       
        bn = f["batch_normalization"]["batch_normalization"]
        model.get_layer(bn_name).set_weights([
            bn["gamma:0"][()],
            bn["beta:0"][()],
            bn["moving_mean:0"][()],
            bn["moving_variance:0"][()]
        ])
        print(f"✅ Loaded: {bn_name}")

       
        h5_dense_keys = ["dense", "dense_1", "dense_2"]
        for live_name, h5_key in zip(dense_names, h5_dense_keys):
            g = f[h5_key][h5_key]
            model.get_layer(live_name).set_weights([
                g["kernel:0"][()],
                g["bias:0"][()]
            ])
            print(f"✅ Loaded: {live_name} ← from h5 key '{h5_key}'")

    return model

try:
    skin_model = load_skin_model()
    print("✅ Skin-Net Online (Phase 2 Fine-Tuned)")
except Exception as e:
    import traceback
    traceback.print_exc()
    print(f"❌ Skin Weight Error: {e}")

EYE_CLASSES = ['Cataract', 'Diabetic Retinopathy', 'Normal']
SKIN_CLASSES = ['Acne', 'Melanoma', 'Scaly_Lesions', 'Vitiligo_Pigmentation', 'Warts_Viral']

TWILIO_SID = "ACb1bdc88d0fcd98d69047e90a92ee5731"
TWILIO_TOKEN ="9194f0faba387348b9d2a9e0c5c1df87"
PUBLIC_URL = "https://diageotropic-donnie-unwatchfully.ngrok-free.dev" 


def analyze_image(img_path, mode):
    try:
        if mode == "EYE":
            img = keras_image.load_img(img_path, target_size=(224, 224))
            img_array = keras_image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            preds = eye_model.predict(img_array)[0]
            classes = EYE_CLASSES
        else:
            img = keras_image.load_img(img_path, target_size=(224, 224))
            img_array = keras_image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = skin_preprocess(img_array) 
            preds = skin_model.predict(img_array)[0]
            classes = SKIN_CLASSES

        idx = np.argmax(preds)
        return classes[idx], preds[idx] * 100
    except Exception as e:
        return f"Error: {e}", 0.0


def process_and_reply_async(data, lang, mode):
    client = Client(TWILIO_SID, TWILIO_TOKEN)
    media_url, user_num, twilio_num = data.get('MediaUrl0'), data.get('From'), data.get('To')

    try:
        response = requests.get(media_url, auth=(TWILIO_SID, TWILIO_TOKEN))
        raw_path = f"temp_{user_num.strip('+')}.jpg"
        with open(raw_path, 'wb') as f: f.write(response.content)
        
        final_img = extract_eye_crop(raw_path) if mode == "EYE" else raw_path
        diagnosis, conf = analyze_image(final_img, mode)
        
        h_path = f"static/heatmap_{user_num.strip('+')}.jpg"
        current_model = eye_model if mode == "EYE" else skin_model
        generate_and_save_heatmap(final_img, current_model, h_path, mode=mode)

        client.messages.create(body=f"🩺 *AI Analysis ({mode}):*\nDetected *{diagnosis}* ({conf:.1f}%).", from_=twilio_num, to=user_num)
        client.messages.create(body="🔍 *AI Heatmap:* Markers visualized.", media_url=[f"{PUBLIC_URL}/{h_path}"], from_=twilio_num, to=user_num)
        
        advice = chatbot_response(f"Explain {diagnosis} and give 2 precautions. Under 80 words.", target_lang=lang)
        client.messages.create(body=f"📝 *AI Advice:* {advice}", from_=twilio_num, to=user_num)
    except Exception as e:
        print(f"❌ Worker Error: {e}")


@app.route("/whatsapp", methods=['POST'])
def whatsapp_reply():
    user_num = request.values.get('From')
    msg = request.values.get('Body', '').strip().lower()
    num_media = int(request.values.get('NumMedia', 0))
    resp = MessagingResponse()

    if msg in ['en', 'te', 'hi']:
        user_languages[user_num] = msg
        resp.message(f"✅ Language: {msg.upper()}. Type 'Eye' or 'Skin' to select scan.")
        return str(resp)
    
    if 'eye' in msg:
        user_modes[user_num] = "EYE"
        resp.message("👁️ *Eye Mode Active.* Send a clear photo.")
        return str(resp)
    if 'skin' in msg:
        user_modes[user_num] = "SKIN"
        resp.message("🩹 *Skin Mode Active.* Send a photo of the lesion.")
        return str(resp)

    if num_media > 0:
        mode = user_modes.get(user_num)
        if not mode:
            resp.message("⚠️ Please type 'Eye' or 'Skin' first!")
        else:
            lang = user_languages.get(user_num, 'en')
            threading.Thread(target=process_and_reply_async, args=(request.values.to_dict(), lang, mode)).start()
            resp.message(f"📸 {mode} Image received! Analyzing...")
        return str(resp)

    ai_msg = chatbot_response(msg, target_lang=user_languages.get(user_num, 'en'))
    resp.message(ai_msg)
    return str(resp)

if __name__ == "__main__":
    app.run(port=5000)