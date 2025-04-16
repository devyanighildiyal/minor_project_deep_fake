import streamlit as st
from PIL import Image
from utils import predict_deepfake

# Color palette
st.markdown("""
    <style>
        .main {
            background-color: #000014;
            color: #f5f5f5;
        }
        .block-container {
            padding-top: 2rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 FakeNet - Deepfake Detection")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Analyze"):
        label, confidence = predict_deepfake(image)
        if label == "FAKE":
            st.error(f"⚠️ Deepfake Detected with {confidence}% confidence")
        else:
            st.success(f"✅ Real Image with {confidence}% confidence")
