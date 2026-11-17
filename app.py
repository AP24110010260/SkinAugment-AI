import streamlit as st
from PIL import Image
from io import BytesIO
from model import load_model, predict_image
from augmentation import generate_augmentations

st.set_page_config(page_title="SkinAugment AI", page_icon="✨", layout="wide")

st.title("SkinAugment AI")
st.subheader("Consumer Skincare Photo Augmenter")
st.write("Generate realistic variations of skincare photographs.")

st.info("This is an image augmentation research tool, not a medical diagnostic system.")

@st.cache_resource
def get_model():
    return load_model()

with st.sidebar:
    st.header("Settings")
    count = st.slider("Number of variations", 1, 10, 5)

uploaded = st.file_uploader("Upload a skincare photograph", type=["jpg", "jpeg", "png"])

if uploaded:
    original = Image.open(uploaded).convert("RGB")
    st.header("Original Image")
    st.image(original, width=400)

    if st.button("Generate Augmented Images", type="primary"):
        st.session_state["variations"] = generate_augmentations(original, count)

    if "variations" in st.session_state:
        st.header("Generated Variations")
        columns = st.columns(3)

        for i, image in enumerate(st.session_state["variations"]):
            with columns[i % 3]:
                st.image(image, caption=f"Variation {i+1}", use_container_width=True)
                buffer = BytesIO()
                image.save(buffer, format="JPEG")
                st.download_button(
                    "Download",
                    buffer.getvalue(),
                    file_name=f"skinaugment_{i+1}.jpg",
                    mime="image/jpeg",
                    key=f"download_{i}"
                )

    st.divider()
    st.header("Image Classification")
    model = get_model()

    if model is None:
        st.warning("Trained model checkpoint not found. Classification will be available after model integration.")
    elif st.button("Classify Original Image"):
        label, confidence, scores = predict_image(model, original)
        st.success(f"Predicted category: {label}")
        st.metric("Model confidence", f"{confidence * 100:.2f}%")
        st.bar_chart(scores)
else:
    st.write("Upload an image to begin.")

st.caption("SkinAugment AI | Image augmentation research project")
