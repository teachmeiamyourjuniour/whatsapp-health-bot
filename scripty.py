import os
import shutil
import random

# --- 1. SET YOUR BASE FOLDER ---
BASE_PATH = "C:\\Users\\Mukundraj\\OneDrive\\Desktop\\f\\skin_disease_model\\data"
# Change this to whatever your messy folder is actually named!
SOURCE_FOLDER_NAME = "train" 

SOURCE_DIR = os.path.join(BASE_PATH, SOURCE_FOLDER_NAME)
TARGET_DIR = os.path.join(BASE_PATH, "skin_dataset_clean")

# --- 2. THE MAPPING (Matches your counts exactly) ---
mapping = {
    "Acne and Rosacea Photos": "Acne",
    "Light Diseases and Disorders of Pigmentation": "Vitiligo_Pigmentation",
    "Melanoma Skin Cancer Nevi and Moles": "Melanoma",
    "Psoriasis&Ringworm and other related diseses": "Scaly_Lesions",
    "Warts Molluscum and other Viral Infections": "Warts_Viral"
}

MAX_PER_CLASS = 450 

if not os.path.exists(TARGET_DIR): 
    os.makedirs(TARGET_DIR)

print(f"🔍 Checking source: {SOURCE_DIR}")

for messy, clean in mapping.items():
    s_path = os.path.join(SOURCE_DIR, messy)
    t_path = os.path.join(TARGET_DIR, clean)
    
    # Check if the folder exists before trying to read it
    if not os.path.exists(s_path):
        print(f"❌ ERROR: Could not find '{messy}'. Skipping...")
        continue
    
    if not os.path.exists(t_path): 
        os.makedirs(t_path)
    
    images = [f for f in os.listdir(s_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    random.shuffle(images)
    
    # Copy files
    for img in images[:MAX_PER_CLASS]:
        shutil.copy(os.path.join(s_path, img), os.path.join(t_path, img))
    
    print(f"✅ Balanced {clean}: {len(images[:MAX_PER_CLASS])} images")

print(f"\n✨ DONE! Clean dataset ready at: {TARGET_DIR}")