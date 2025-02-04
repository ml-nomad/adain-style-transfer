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
To perform style transfer, upload a content image and a style image
""")

content_file = st.file_uploader("Content Image", type=["jpg", "jpeg", "png"])
style_file = st.file_uploader("Style Image", type=["jpg", "jpeg", "png"])

# Create two columns for displaying images with equal width
if content_file or style_file:
    col1, col_space, col2 = st.columns([1, 0.2, 1])

    with col1:
        if content_file:
            content = Image.open(content_file).convert("RGB")
            st.image(content, caption="Content", use_container_width=True)

    with col2:
        if style_file:
            style = Image.open(style_file).convert("RGB")
            st.image(style, caption="Style", use_container_width=True)

if st.button("Run"):
    if not content_file or not style_file:
        st.warning("Both content and style images are required")
    else:
        content = Image.open(content_file).convert("RGB")
        style = Image.open(style_file).convert("RGB")

        # Add spinner while processing
        with st.spinner("Generating stylized image..."):
            result = transfer_style_img(model, content, style, get_device())

        # Display result after processing is complete
        st.subheader("Result")
        st.image(result, use_container_width=True)