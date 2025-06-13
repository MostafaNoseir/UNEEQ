import streamlit as st
import joblib

# Load trained pipeline
pipeline = joblib.load("sentiment_classifier_pipeline.pkl")

label_map = {-1: "Negative", 0: "Neutral", 1: "Positive"}
color_map = {-1: "red", 0: "gray", 1: "green"}

st.title("🧠 Sentiment Classifier")
st.markdown("""
This simple web app analyzes a piece of text and predicts whether the sentiment is **Positive**, **Neutral**, or **Negative**.
It uses a machine learning model with natural language preprocessing to understand the meaning of your text.
""")

user_input = st.text_area("Enter a text to classify its sentiment:", height=150)

if st.button("Predict"):
    if user_input.strip():
        prediction = pipeline.predict([user_input])[0]

        # Custom styled result box
        st.markdown(
            f"<h4 style='color:{color_map[prediction]};'>Predicted Sentiment: {label_map[prediction]}</h4>",
            unsafe_allow_html=True
        )
    else:
        st.warning("Please enter some text.")