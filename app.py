import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Pothole Classifier", layout="centered")
st.title("🕳️ Pothole Image Classification with CNN")

# Try loading the model
try:
    model = load_model('baseline_cnn_model.h5')
    st.success("✅ Model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load model: {e}")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    try:
        image = Image.open(uploaded_file).resize((128, 128))
        img_array = np.array(image) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)[0][0]
        class_label = '🟢 Class 1 (Pothole)' if prediction > 0.5 else '🔵 Class 0 (No Pothole)'

        st.subheader(f"Prediction: {class_label}")
        st.write(f"🧠 Confidence: `{prediction:.2f}`")
    except Exception as e:
        st.error(f"❌ Error processing image: {e}")
else:
    st.info("👈 Please upload a .jpg or .png image to classify.")
