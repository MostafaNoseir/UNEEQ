import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

# --- Set page config ---
st.set_page_config(page_title="Digit Classifier", layout="centered")

# --- Load the trained model ---
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("cnn_model.keras")

model = load_model()

# --- Helper function to preprocess the image ---
def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to 28x28
    image_array = np.array(image) / 255.0  # Normalize
    image_array = image_array.reshape(1, 28, 28, 1)  # Reshape for model
    return image_array

# --- Title and Description ---
st.title("🧠 Handwritten Digit Classifier")
st.markdown("""
Upload an image of a digit (28x28 grayscale). The model will predict which digit (0–9) it is.

*Powered by a Convolutional Neural Network trained on MNIST.*
""")

# --- File uploader ---
uploaded_file = st.file_uploader("📤 Upload a digit image", type=["png", "jpg", "jpeg"])

# --- Sidebar ---
with st.sidebar:
    st.markdown("## ℹ️ Model Info")
    st.write("**Architecture:** CNN")
    st.write("**Input Shape:** (28, 28, 1)")
    st.write("**Output Classes:** 10 (digits 0–9)")

# --- Main logic ---
if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=150)

        input_data = preprocess_image(image)

        prediction = model.predict(input_data)[0]
        predicted_class = int(np.argmax(prediction))

        st.markdown(f"## ✅ Predicted Digit: **{predicted_class}**")

        # Show prediction probabilities
        fig, ax = plt.subplots()
        bars = ax.bar(range(10), prediction, color="skyblue")
        ax.set_xticks(range(10))
        ax.set_xlabel("Digit")
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Confidence")

        for i, bar in enumerate(bars):
            height = bar.get_height()
            ax.text(i, height + 0.01, f"{height:.2f}", ha='center', va='bottom', fontsize=8)

        st.pyplot(fig)

    except Exception as e:
        st.error(f"⚠️ Error processing the image: {str(e)}")

else:
    st.info("Please upload a digit image to get started.")