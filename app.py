import streamlit as st
import joblib
import numpy as np
from PIL import Image, ImageOps

# Load the trained Random Forest model
model = joblib.load("random_forest_mnist.pkl")

st.set_page_config(page_title="Digit Classifier", page_icon="✍️")
st.title("🧠 Handwritten Digit Classifier")

st.markdown("""
Upload an image of a **handwritten digit** (0-9).  
Make sure it's **centered and white on black background** like MNIST format.
""")

# File uploader
uploaded_file = st.file_uploader("Choose a digit image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('L')  # Convert to grayscale

    # Resize and invert (MNIST-style)
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    st.image(image, caption='Processed Image', width=150)

    # Preprocess for model
    img_array = np.array(image).astype('float32') / 255.0
    img_flattened = img_array.reshape(1, -1)

    # Predict
    prediction = model.predict(img_flattened)[0]
    st.success(f"✅ Predicted Digit: **{prediction}**")
