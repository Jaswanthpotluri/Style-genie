#  Style Genie – Intelligent Skin Tone Classification & Outfit Recommendation System

**Style Genie** is a deep learning–driven application that intelligently detects a user's skin tone from facial images and generates personalized fashion suggestions using **Google Gemini Pro (Generative AI)**. This project integrates computer vision, transfer learning, and generative text modeling to provide a seamless AI fashion assistant experience.

---

Dataset: The Monk Skin Tone (MST) Scale is a 10-shade scale developed by Dr. Ellis Monk in partnership with Google to represent a broader range of human skin tones than previously available scales. https://skintone.google/mste-dataset
<img width="1576" height="236" alt="image" src="https://github.com/user-attachments/assets/5b420d99-7821-4bec-b03c-3f642439e959" />

## 🧠 What It Does

- 📸 **Skin Tone Classification**  
  Uses facial image input to classify skin tone based on the **Monk Skin Tone (MST)** scale.

- 🎯 **Face Preprocessing**  
  Applies **MTCNN** to detect the face, then enhances it with **CLAHE** and **Cr-Cb skin masking** for more reliable predictions under varied lighting.

- 🔍 **ResNet50-based CNN Model**  
  Fine-tuned on labeled face images with brightness-aware augmentations to achieve robust classification (~97% accuracy).

- 👗 **Outfit Recommendations via Gemini**  
  Once a skin tone is identified, the system queries **Google Gemini 1.5 Pro** to generate stylish, context-aware dress color suggestions based on the user's skin tone and the given occasion.

---

## 🚀 Key Highlights

✅ Robust face detection & enhancement pipeline  
✅ Custom-trained ResNet50 model on real-world face data  
✅ Gemini API integration for dynamic, human-like fashion advice  
✅ GUI for easy image upload (Tkinter-based)  
✅ Modular code structure with separate training, testing, and API layers

---

## 🔧 Tools & Technologies

- **Deep Learning**: TensorFlow, Keras, ResNet50  
- **Image Processing**: OpenCV, CLAHE, HSV/CrCb masking  
- **Face Detection**: MTCNN (Multi-task Cascaded CNN)  
- **Generative AI**: Google Gemini 1.5 Pro API  
- **Interface**: Tkinter for user-friendly image upload  
- **Others**: Scikit-learn, NumPy, pandas, logging

---

## 🧪 Example Workflow

1. User uploads a facial image via GUI  
2. MTCNN detects and crops the face  
3. Image is enhanced using CLAHE and masked for skin regions  
4. The CNN model classifies the image into an MST skin tone label  
5. User inputs an occasion (e.g., casual, wedding, party)  
6. Gemini Pro returns stylish outfit color suggestions tailored to both tone & context

---

## 📈 Model Performance

- **Accuracy**: ~97% on test split  
- **Input Resolution**: 224×224 RGB  
- **Labels**: Monk Skin Tone Scale (encoded using LabelEncoder)  
- **Augmentations**: Brightness shifts, rotations, flips for better lighting robustness

---

## 🔮 Why It Matters

Style Genie demonstrates how AI can bridge **computer vision** and **personal style**, delivering practical and intelligent user experiences. It also showcases the power of combining **discriminative models (CNN)** with **generative models (LLMs)** in a real-world application.

This project is ideal for fashion tech applications, virtual try-ons, smart mirrors, and personalized recommendation systems.

