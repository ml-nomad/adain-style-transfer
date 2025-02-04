import streamlit as st
from PIL import Image

from src.transfer_style import transfer_style_img
from src.utils import get_device, load_saved_model


@st.cache_resource
def load_model():
    device = get_device()
    model = load_saved_model("model/model.pth", device)

    return model

model = load_model()

st.title("Deep Learning School training project")

st.write("""
This demo app performs arbitrary style transfer using a pre-trained model.
To perform style transfer, upload a content image and a style image, and adjust Style intensity parameter to control style strength
""")

content_file = st.file_uploader("Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Style Image", type=["jpg", "jpeg", "png"])

alpha = st.slider("Style intensity", 0.0, 1.0, 1.0, 0.1)

if st.button("Run"):
    if not content_file or not style_file:
        st.warning("Both content and style images are required")
    else:
        content = Image.open(content_file).convert("RGB")
        style = Image.open(style_file).convert("RGB")

        result = transfer_style_img(model, content, style, get_device(), alpha)

        st.image([content, style], caption=["Content", "Style"], width=300)

        # Отображаем результат
        st.subheader("Result")
        st.image(result, use_container_width=True)