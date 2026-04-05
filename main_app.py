import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from medical_bot import chatbot_response # Importing your bot logic!

# 1. LOAD THE "EYES" (CNN Model)
print("--- INITIALIZING EYE DISEASE MODEL ---")
model = load_model('eye_disease_model.h5')
class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

def run_diagnosis(img_path):
    # Preprocess
    img = cv2.imread(img_path)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)
    
    # Predict
    preds = model.predict(img_array)
    result = class_names[np.argmax(preds)]
    confidence = np.max(preds) * 100
    return result, confidence

# --- START THE MULTIMODAL FLOW ---
image_to_test = "dataset/cataract/0_left.jpg" # Example image
diagnosis, conf = run_diagnosis(image_to_test)

print(f"\n📢 DIAGNOSIS: {diagnosis} ({conf:.2f}%)")

# 2. AUTOMATIC CONSULTATION
print(f"--- AI CONSULTANT STARTING ---")
initial_query = f"I have been diagnosed with {diagnosis}. What does this mean and what are the next steps?"
ai_advice = chatbot_response(initial_query, target_lang='te') # Defaulting to Telugu for demo!

print(f"\nAI ADVICE (Telugu):\n{ai_advice}")