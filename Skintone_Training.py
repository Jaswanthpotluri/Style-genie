import os
import numpy as np
import pandas as pd
import gc
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input, ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle

# Paths (Update if needed)
processed_folder = r"C:\Users\potlu\PycharmProjects\pythonProject2\processed_images"
csv_path = r"C:\Users\potlu\PycharmProjects\pythonProject2\rangu1.csv"

# Load CSV
df = pd.read_csv(csv_path)

# Load processed images and labels
processed_faces = []
labels = []

for index, row in df.iterrows():
    image_name = row['image_ID']
    skin_tone = row['MST']

    image_path = os.path.join(processed_folder, image_name)
    if not os.path.exists(image_path):
        print(f"⚠ Warning: {image_path} not found. Skipping...")
        continue

    image = cv2.imread(image_path)

    # Convert to YCrCb for CLAHE
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)

    # Apply CLAHE on the luminance channel
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    y = clahe.apply(y)
    ycrcb = cv2.merge([y, cr, cb])
    image = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)

    # Skin masking (CrCb thresholding)
    ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    lower = np.array([0, 133, 77], dtype=np.uint8)
    upper = np.array([255, 173, 127], dtype=np.uint8)
    mask = cv2.inRange(ycrcb, lower, upper)

    # Apply the mask
    image = cv2.bitwise_and(image, image, mask=mask)

    # Resize and convert
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    processed_faces.append(img_to_array(image).astype('float32'))
    labels.append(skin_tone)

# Convert to NumPy arrays
X = np.array(processed_faces, dtype="float32")
y_skin_tone = np.array(labels)

del processed_faces
gc.collect()

# Preprocess and encode
X = preprocess_input(X)

label_encoder_skin2 = LabelEncoder()
y_skin_tone_encoded = label_encoder_skin2.fit_transform(y_skin_tone)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y_skin_tone_encoded, test_size=0.2, random_state=42)

del X, y_skin_tone_encoded
gc.collect()

# Image augmentation
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)

train_generator = datagen.flow(X_train, y_train, batch_size=4)

# Load ResNet50 base
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

for layer in base_model.layers:
    layer.trainable = False

# Custom classifier
x = Flatten()(base_model.output)
x = Dense(512, activation="relu", kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
output = Dense(len(label_encoder_skin2.classes_), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=output)

# Unfreeze last 50 layers
for layer in base_model.layers[-50:]:
    layer.trainable = True

# Compile model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# Define Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, verbose=1)

# Train
model.fit(train_generator, validation_data=(X_test, y_test),
          epochs=20, batch_size=4, callbacks=[early_stop, reduce_lr])

# Save model
model.save("skin_tone_model_v4.keras")

# Save LabelEncoder to disk
with open("venv/label_encoder_skin2.pkl", "wb") as f:
    pickle.dump(label_encoder_skin2, f)