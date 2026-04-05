import cv2
import mediapipe as mp
import numpy as np
import os

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True, 
    max_num_faces=1, 
    refine_landmarks=True
)

def extract_eye_crop(image_path, output_path="static/processed_eye.jpg"):
    """
    Function to find, crop and save the eye region.
    """
    try:
        img = cv2.imread(image_path)
        if img is None: 
            print(f"❌ Error: Could not read image at {image_path}")
            return image_path
        
        h, w, _ = img.shape
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_img)
        
        if not results.multi_face_landmarks:
            print("⚠️ No eye detected. Using original image.")
            return image_path 

        landmarks = results.multi_face_landmarks[0].landmark
        
        eye_indices = [33, 133, 159, 145, 153, 154, 155] 
        
        x_coords = [int(landmarks[i].x * w) for i in eye_indices]
        y_coords = [int(landmarks[i].y * h) for i in eye_indices]
        
        xmin, xmax = min(x_coords), max(x_coords)
        ymin, ymax = min(y_coords), max(y_coords)
        
       
        width_b = int((xmax - xmin) * 0.6)
        height_b = int((ymax - ymin) * 0.6)
        
        y_start, y_end = max(0, ymin-height_b), min(h, ymax+height_b)
        x_start, x_end = max(0, xmin-width_b), min(w, xmax+width_b)
        
        crop = img[y_start:y_end, x_start:x_end]
        
        if not os.path.exists('static'): 
            os.makedirs('static')
            
        cv2.imwrite(output_path, crop)
        print(f"✅ Success: Eye cropped to {output_path}")
        return output_path
        
    except Exception as e:
        print(f"❌ Cropper logic error: {e}")
        return image_path