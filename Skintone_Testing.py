import cv2
import numpy as np
from mtcnn import MTCNN
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import pickle
from tkinter import Tk, filedialog

# === Load Model & Label Encoder ===
model = load_model("skin_tone_model_v4.keras")
with open("label_encoder_skin2.pkl", "rb") as f:
    label_encoder_skin = pickle.load(f)

# === Apply CLAHE ===
def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    merged = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)
    return enhanced_image

# === Apply Skin Mask ===
def apply_skin_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    skin = cv2.bitwise_and(image, image, mask=mask)
    return skin

# === Face Detection and Preprocessing ===
def detect_and_crop_face(image_path):
    detector = MTCNN()
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found or invalid format.")
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    faces = detector.detect_faces(image_rgb)
    if not faces:
        raise ValueError("No face detected!")

    x, y, w, h = faces[0]['box']
    x, y = max(0, x), max(0, y)
    cropped_face = image_rgb[y:y+h, x:x+w]
    cropped_face = cv2.resize(cropped_face, (224, 224))

    # Apply CLAHE and Skin Mask
    clahe_applied = apply_clahe(cropped_face)
    skin_masked = apply_skin_mask(clahe_applied)
    return skin_masked

# === Skin Tone Prediction ===
def predict_skin_tone(image_path):
    try:
        processed_face = detect_and_crop_face(image_path)
        face_array = img_to_array(processed_face).astype('float32')
        face_array = preprocess_input(face_array)
        face_array = np.expand_dims(face_array, axis=0)

        prediction = model.predict(face_array)
        predicted_label_index = np.argmax(prediction)
        predicted_label = label_encoder_skin.inverse_transform([predicted_label_index])[0]

        print(f"Predicted Skin Tone: {predicted_label}")
    except Exception as e:
        print(f"Error: {e}")

# === Select Image from File Dialog ===
def select_image():
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        predict_skin_tone(file_path)
    else:
        print("No file selected.")

# === Run ===
if __name__ == "__main__":
    select_image()