import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import boto3

# AWS S3 Configuration
S3_BUCKET_NAME = "suprith0230"
S3_MODEL_PATH = "wildlife_model.h5"
LOCAL_MODEL_PATH = "wildlife_model.h5"

# Initialize S3 Client
s3 = boto3.client("s3")

# Download model if not present
try:
    s3.download_file(S3_BUCKET_NAME, S3_MODEL_PATH, LOCAL_MODEL_PATH)
    st.success("Model downloaded successfully from S3!")
except Exception as e:
    st.error(f"Error downloading model: {e}")

# Load model
model = tf.keras.models.load_model(LOCAL_MODEL_PATH)

species_list = ["Lion", "Coyote", "Elephant"]

st.title("WildTrack AI: Wildlife Footprint Classification")

uploaded_file = st.file_uploader("Upload a footprint image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Preprocess the image
    image = np.array(image)
    image = cv2.resize(image, (128, 128)) / 255.0
    image = np.expand_dims(image, axis=0)
    
    # Predict species
    prediction = model.predict(image)
    predicted_species = species_list[np.argmax(prediction)]
    confidence = np.max(prediction)

    st.write(f"Predicted Species: **{predicted_species}**")
    st.write(f"Confidence Score: **{confidence:.2f}**")
