import streamlit as st
import numpy as np
from PIL import Image
import io
import cv2
import chromakey
from scipy.ndimage import distance_transform_edt

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
    erode_px = st.sidebar.slider("Fringe removal strength", 0, 5, 2, help="How many pixels in from the edge are treated as 'trusted' garment color, which then gets grown back outward to replace green-tinted spill pixels. Raise if a green rim persists; lower if fine details look smeared.")

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
                alpha_f = 1 - mask  # 0..1, foreground=1

                # Decontamination alone still leaves a faint green rim on real photos
                # (uneven studio lighting, anti-aliased edges in the source image).
                # Instead of shrinking the silhouette (which loses shape on thin details),
                # mark a "trusted core" a few pixels in from the edge as clean garment
                # color, then grow that color outward to replace the green-tinted edge
                # pixels — while keeping the ORIGINAL soft alpha, so the outline shape
                # and antialiasing are preserved.
                final_rgb = out
                if erode_px > 0:
                    fg_binary = (alpha_f > 0.5).astype(np.uint8)
                    kernel = np.ones((3, 3), np.uint8)
                    core = cv2.erode(fg_binary, kernel, iterations=erode_px)

                    if core.any():
                        # For every non-core pixel, find the nearest core pixel and
                        # copy its color outward (nearest-neighbor color extension).
                        _, indices = distance_transform_edt(1 - core, return_indices=True)
                        grown_rgb = out[indices[0], indices[1]]
                        final_rgb = np.where(core[..., None].astype(bool), out, grown_rgb)

                alpha = np.uint8(np.clip(alpha_f, 0, 1) * 255)
                rgba = np.dstack([final_rgb, alpha])
                extracted_image = Image.fromarray(rgba.astype(np.uint8), "RGBA")

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
