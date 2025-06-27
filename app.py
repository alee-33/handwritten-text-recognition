import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load trained model
# model = tf.keras.models.load_model("digit_model.h5")
model = tf.keras.models.load_model(r"C:\Users\Ali\Desktop\cvproc\AI_LAB_GROUP_TASK_[25,26,33,42,58]\digit_model.h5")



# Preprocessing function
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = ImageOps.invert(image)  # Invert so black digit is on white background
    image = image.resize((28, 28))  # Resize to 28x28
    image = np.array(image) / 255.0  # Normalize
    image = image.reshape(1, 28, 28, 1)  # Reshape for model
    return image


st.title("🖋️ Handwritten Digit Recognizer")

uploaded_file = st.file_uploader("Upload an image of a digit (0–9)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        input_data = preprocess_image(img)
        prediction = model.predict(input_data)
        digit = np.argmax(prediction)
        st.success(f"Predicted Digit: {digit}")
