import streamlit as st
import numpy as np
from PIL import Image, ImageOps
import joblib
import os

# App title
st.set_page_config(page_title="Digit Classifier", page_icon="✍️")
st.title("Handwritten Digit Classifier")

st.markdown("""
Upload an image of a handwritten digit (28x28 pixels, grayscale). The model will predict the digit using Logistic Regression.
""")

MODEL_PATH = "logistic_model.pkl"

# Load model
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found! Please ensure 'logistic_model.pkl' is present.")
    st.stop()

model = joblib.load(MODEL_PATH)

# Upload image
uploaded_file = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    # Preprocess the image
    image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)                 # Invert image (black background to white digit)
    image = image.resize((28, 28))                 # Resize to MNIST format

    st.image(image, caption="Processed Image", width=150)

    # Convert image to input format
    img_array = np.array(image).astype("float32") / 255.0
    img_flattened = img_array.reshape(1, -1)

    # Predict
    prediction = model.predict(img_flattened)[0]
    st.success(f" Predicted Digit: **{prediction}**")
