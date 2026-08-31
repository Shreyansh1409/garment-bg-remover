import streamlit as st
import numpy as np
from PIL import Image
import io
import chromakey

st.set_page_config(layout="wide", page_title="Outfit Extractor")
st.title("🟩 Outfit / Garment Extractor")
st.write("Upload a green screen product photo to isolate the garment using chroma keying.")

st.sidebar.header("Chroma Key Settings")
key_color_hex = st.sidebar.color_picker("Pick your background color", "#00FF00")
tola = st.sidebar.slider("Edge softness (tola)", 1, 50, 10, help="Distance below which a pixel is fully kept as foreground.")
tolb = st.sidebar.slider("Edge softness (tolb)", tola + 1, 100, 30, help="Distance above which a pixel is fully treated as background.")

uploaded_file = st.file_uploader("Upload your green screen photo", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    original_image = Image.open(uploaded_file).convert("RGB")

    with col1:
        st.header("Original Image")
        st.image(original_image, width="stretch")

    with col2:
        st.header("Extracted Garment")
        with st.spinner("Calculating chroma key..."):
            try:
                img_array = np.array(original_image)  # RGB, shape (H, W, 3)

                # chroma_key() takes a hex string, returns (out, mask) — NOT a single RGBA array.
                # out = RGB with the key color subtracted (can look dull/off-color at edges)
                # mask = float array, 1.0 = background, 0.0 = foreground
                out, mask = chromakey.chroma_key(
                    img_array, key_color_hex, tola=tola, tolb=tolb
                )

                # Build alpha from the mask (invert: foreground = opaque) and composite
                # onto the ORIGINAL pixels, not `out`, to avoid the color-subtracted look.
                alpha = np.uint8((1 - mask) * 255)
                rgba = np.dstack([img_array, alpha])
                extracted_image = Image.fromarray(rgba, "RGBA")

                st.image(extracted_image, width="stretch")

                buf = io.BytesIO()
                extracted_image.save(buf, format="PNG")
                byte_im = buf.getvalue()

                st.download_button(
                    label="Download Transparent PNG",
                    data=byte_im,
                    file_name="chromakey_cutout.png",
                    mime="image/png",
                    width="stretch",
                )
            except Exception as e:
                st.error(f"Error processing image: {e}")
