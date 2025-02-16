import gdown
import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import os

# Download model from Google Drive
file_id = "15kkzDveztNBZpL-w5A7kRoegmStWsV_J"
model_path = "wildlife_model.h5"

if not os.path.exists(model_path):  # Avoid re-downloading
    gdown.download(f"https://drive.google.com/uc?id={file_id}", model_path, quiet=False)

# Load trained model
model = tf.keras.models.load_model(model_path)

# Updated species list
species_list = ["Coyote", "Elephant", "Lion"]

st.title("WildTrack AI: Wildlife Footprint Classification")

uploaded_file = st.file_uploader("Upload a footprint image", type=["jpg", "jpg", "jpg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess the image
    image = np.array(image)
    image = cv2.resize(image, (128, 128)) / 255.0  # Resize and normalize
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Predict species
    prediction = model.predict(image)
    predicted_species = species_list[np.argmax(prediction)]
    confidence = np.max(prediction)

    prediction = model.predict(image)
    print("Raw predictions:", prediction)  # Debugging step

    st.write(f"Predicted Species: **{predicted_species}**")
    st.write(f"Confidence Score: **{confidence:.2f}**")
