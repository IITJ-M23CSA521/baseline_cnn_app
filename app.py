import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

st.set_page_config(page_title="Pothole Classifier", layout="centered")
st.title("🕳️ Pothole Image Classification with CNN & Custom CCT")

# Load both models
try:
    baseline_model = load_model('baseline_cnn_model.h5')
    st.success("✅ Baseline CNN model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load baseline CNN model: {e}")

try:
    custom_model = load_model('custom_cct_model.h5')
    st.success("✅ Custom CCT model loaded successfully.")
except Exception as e:
    st.error(f"❌ Failed to load custom CCT model: {e}")

# File uploader
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    try:
        # Preprocess for baseline CNN model (128x128)
        img_128 = Image.open(uploaded_file).resize((128, 128))
        img_array_128 = np.array(img_128) / 255.0
        img_array_128 = np.expand_dims(img_array_128, axis=0)

        # Preprocess for custom CCT model (224x224)
        img_224 = Image.open(uploaded_file).resize((224, 224))
        img_array_224 = np.array(img_224) / 255.0
        img_array_224 = np.expand_dims(img_array_224, axis=0)

        # Baseline model prediction
        baseline_pred = baseline_model.predict(img_array_128)[0][0]
        baseline_label = '🟢 Pothole' if baseline_pred > 0.5 else '🔵 No Pothole'

        # Custom model prediction
        custom_pred = custom_model.predict(img_array_224)[0][0]
        custom_label = '🟢 Pothole' if custom_pred > 0.5 else '🔵 No Pothole'

        st.markdown("### 📊 Prediction Results")
        st.write(f"**🧠 Baseline CNN Model**: {baseline_label} (Confidence: `{baseline_pred:.2f}`)")
        st.write(f"**🧪 Custom CCT Model**: {custom_label} (Confidence: `{custom_pred:.2f}`)")

    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")
else:
    st.info("👈 Please upload a .jpg or .png image to classify.")
