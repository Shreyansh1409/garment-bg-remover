import streamlit as st
import numpy as np
from PIL import Image
import io
import chromakey

st.set_page_config(layout="wide", page_title="Outfit Extractor")
st.title("🟩 Outfit / Garment Extractor")
st.write("Upload a green screen product photo to isolate the garment using chroma keying.")


def detect_background_hex(img_array: np.ndarray, border: int = 10) -> str:
    """Average the pixels along the image border (almost always background)
    and return their color as a hex string, so the key color picker starts
    on the actual background shade instead of a hardcoded pure green."""
    h, w, _ = img_array.shape
    border = min(border, h // 2, w // 2) or 1
    samples = np.concatenate([
        img_array[:border, :, :].reshape(-1, 3),
        img_array[-border:, :, :].reshape(-1, 3),
        img_array[:, :border, :].reshape(-1, 3),
        img_array[:, -border:, :].reshape(-1, 3),
    ])
    avg = samples.mean(axis=0).astype(np.uint8)
    return "#{:02X}{:02X}{:02X}".format(*avg)


uploaded_file = st.file_uploader("Upload your green screen photo", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    detected_hex = detect_background_hex(np.array(original_image))

    st.sidebar.header("Chroma Key Settings")
    st.sidebar.caption(f"Detected background color: {detected_hex}")
    key_color_hex = st.sidebar.color_picker("Background color", detected_hex)
    tola = st.sidebar.slider("Edge softness (tola)", 1, 50, 10, help="Distance below which a pixel is fully kept as foreground.")
    tolb = st.sidebar.slider("Edge softness (tolb)", tola + 1, 120, 60, help="Distance above which a pixel is fully treated as background. Raise this if uneven lighting leaves background pixels partially opaque.")

    col1, col2 = st.columns(2)

    with col1:
        st.header("Original Image")
        st.image(original_image, width="stretch")

    with col2:
        st.header("Extracted Garment")
        with st.spinner("Calculating chroma key..."):
            try:
                img_array = np.array(original_image)  # RGB, shape (H, W, 3)

                # chroma_key() takes a hex string, returns (out, mask) — NOT a single RGBA array.
                # out = RGB with the key color subtracted from every pixel in proportion to
                #       the mask — this "decontaminates" green spill at semi-transparent
                #       edges (anti-aliased pixels blended with the background in the
                #       original photo). Fully-opaque interior pixels are unaffected.
                # mask = float array, 1.0 = background, 0.0 = foreground
                out, mask = chromakey.chroma_key(
                    img_array, key_color_hex, tola=tola, tolb=tolb
                )

                # Use `out` (spill-decontaminated) for RGB, not the raw original — the raw
                # pixels still carry a green tint at edges even where alpha is partial,
                # which shows up as a green fringe once composited on any other background.
                alpha = np.uint8((1 - mask) * 255)
                rgba = np.dstack([out, alpha])
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
