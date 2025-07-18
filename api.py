import cv2
import numpy as np
import pickle
import logging
import google.generativeai as genai
from tkinter import Tk, filedialog
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# ========== SETUP LOGGING ==========
logging.basicConfig(filename='style_genie_deep.log', level=logging.INFO)

# ========== GEMINI API CONFIG ==========
GENAI_API_KEY = "here put your api key"
genai.configure(api_key=GENAI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-pro-latest")

# ========== LOAD MODEL & ENCODER ==========
cnn_model = load_model("skin_tone_model_v4.keras")
with open("label_encoder_skin2.pkl", "rb") as f:
    skin_label_encoder = pickle.load(f)

# ========== IMAGE ENHANCEMENT FUNCTIONS ==========
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_image = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced_image, cv2.COLOR_LAB2RGB)

def apply_skin_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    return cv2.bitwise_and(image, image, mask=mask)

# ========== FACE DETECTION & PREPROCESSING ==========
def detect_and_crop_face(image_path):
    detector = MTCNN()
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid format.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    if not faces:
        raise ValueError("No face detected.")
    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)
    face_crop = image_rgb[y:y+h, x:x+w]
    face_crop = cv2.resize(face_crop, (224, 224))
    face_clahe = apply_clahe(face_crop)
    skin_only = apply_skin_mask(face_clahe)
    return skin_only

# ========== SKIN TONE PREDICTION ==========
def predict_skin_tone(image_path):
    try:
        processed_face = detect_and_crop_face(image_path)
        face_array = img_to_array(processed_face).astype('float32')
        face_array = preprocess_input(face_array)
        face_array = np.expand_dims(face_array, axis=0)

        predictions = cnn_model.predict(face_array)
        predicted_index = np.argmax(predictions)
        predicted_label = skin_label_encoder.inverse_transform([predicted_index])[0]
        confidence = predictions[0][predicted_index]

        return predicted_label, confidence
    except Exception as e:
        logging.error(f"Prediction Error: {str(e)}")
        return None, None

# ========== GEMINI FASHION SUGGESTION ==========
def get_dress_suggestion(skin_tone, occasion):
    prompt = f"Suggest stylish dress colors and outfit ideas for a person with {skin_tone} skin tone for a {occasion} occasion."
    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "No suggestion generated."
    except Exception as e:
        logging.error(f"Gemini Error: {str(e)}")
        return "Gemini API error while generating suggestion."

# ========== UPLOAD IMAGE ==========
def upload_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Your Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    return file_path if file_path else None

# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    print("\n🌟 Welcome to Style Genie – Powered by Deep Learning + Gemini 🌟")

    img_path = upload_image()
    if img_path:
        print("🖼 Image uploaded successfully. Analyzing skin tone...")
        skin_tone, confidence = predict_skin_tone(img_path)
        if skin_tone:
            print(f"\n✅ Detected Skin Tone: {skin_tone} ({confidence:.2%} confidence)")
            occasion = input("\n💬 Enter the occasion (e.g., wedding, casual, party): ").strip()
            suggestion = get_dress_suggestion(skin_tone, occasion)
            print("\n👗 Dress Suggestion:")
            print(suggestion)
        else:
            print("❌ Failed to predict skin tone. Please try with a clearer face image.")
    else:
        print("⚠ No image selected. Please upload a valid image.")