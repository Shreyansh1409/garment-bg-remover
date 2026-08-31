import streamlit as st
import numpy as np
from PIL import Image
import io
import cv2
import chromakey
from scipy.ndimage import distance_transform_edt

st.set_page_config(layout="wide", page_title="Cutaway — Garment Extractor", page_icon="✂️")

# ---------------------------------------------------------------------------
# Styling
#
# Visual language borrows from the pattern-cutting table: a cutting-mat green
# workspace, paper swatch cards for the images, a bobbin-thread red for
# actions, and a brass-pin gold for secondary accents. Two typefaces: an
# editorial serif (Fraunces) for the masthead, a plain grotesk (Inter) for
# every working label, control, and button.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Inter:wght@400;500;600&display=swap');

    :root {
        --ink: #1E2A22;
        --panel: #24332A;
        --paper: #F4EFE2;
        --paper-dim: #E9E1CC;
        --thread: #B23A2E;
        --thread-dark: #8F2C22;
        --brass: #C79A4B;
        --text-on-ink: #F1ECDD;
        --text-on-paper: #23281F;
        --line-on-ink: rgba(241, 236, 221, 0.16);
        --line-on-paper: rgba(35, 40, 31, 0.14);
    }

    html, body, [data-testid="stAppViewContainer"] {
        background-color: var(--ink);
        color: var(--text-on-ink);
        font-family: 'Inter', sans-serif;
    }

    [data-testid="stHeader"] { background-color: transparent; }

    [data-testid="stSidebar"] {
        background-color: var(--panel);
        border-right: 1px solid var(--line-on-ink);
    }
    [data-testid="stSidebar"] * { color: var(--text-on-ink) !important; }

    .cw-hero {
        background-image:
            repeating-linear-gradient(0deg, var(--line-on-ink) 0 1px, transparent 1px 32px),
            repeating-linear-gradient(90deg, var(--line-on-ink) 0 1px, transparent 1px 32px);
        border-bottom: 1px solid var(--line-on-ink);
        padding: 2.75rem 0.5rem 2.25rem 0.5rem;
        margin: -1rem -1rem 2rem -1rem;
        text-align: left;
    }
    .cw-hero h1 {
        font-family: 'Fraunces', serif;
        font-weight: 600;
        font-size: 2.75rem;
        letter-spacing: -0.01em;
        margin: 0 0 0.4rem 0;
        color: var(--text-on-ink);
    }
    .cw-hero p {
        font-size: 1.02rem;
        max-width: 46ch;
        color: var(--text-on-ink);
        opacity: 0.78;
        margin: 0 0 0.9rem 0;
    }
    .cw-stitch {
        border: none;
        border-top: 2px dashed var(--brass);
        width: 64px;
        margin: 0 0 0.9rem 0;
        opacity: 0.85;
    }

    /* Sidebar heading */
    .cw-sidebar-title {
        font-family: 'Fraunces', serif;
        font-size: 1.15rem;
        font-weight: 600;
        margin-bottom: 0.15rem;
    }
    .cw-sidebar-sub {
        font-size: 0.82rem;
        opacity: 0.65;
        margin-bottom: 1.1rem;
    }

    /* Swatch card label chips above each image */
    .cw-chip {
        display: inline-block;
        font-size: 0.78rem;
        font-weight: 500;
        letter-spacing: 0.02em;
        color: var(--text-on-paper);
        background: var(--brass);
        padding: 0.2rem 0.6rem;
        border-radius: 3px;
        margin-bottom: 0.6rem;
    }

    /* Paper cards for the two image columns */
    [data-testid="column"] > div {
        background-color: var(--paper);
        border: 1px solid var(--line-on-paper);
        border-radius: 6px;
        padding: 1.1rem 1.1rem 1.4rem 1.1rem;
    }
    [data-testid="column"] * :not(.cw-chip) { color: var(--text-on-paper); }
    [data-testid="stImage"] img {
        border-radius: 3px;
    }

    /* Buttons */
    .stButton > button, [data-testid="stDownloadButton"] button {
        background-color: var(--thread);
        color: var(--text-on-ink) !important;
        border: none;
        border-radius: 4px;
        padding: 0.55rem 1.1rem;
        font-weight: 500;
        transition: background-color 0.15s ease;
    }
    .stButton > button:hover, [data-testid="stDownloadButton"] button:hover {
        background-color: var(--thread-dark);
    }
    .stButton > button:focus-visible, [data-testid="stDownloadButton"] button:focus-visible {
        outline: 2px solid var(--brass);
        outline-offset: 2px;
    }

    /* Sliders + color picker accent */
    [data-testid="stSlider"] [role="slider"] { background-color: var(--thread) !important; }
    [data-testid="stSlider"] div[style*="background-color: rgb(255, 75, 75)"] { background-color: var(--thread) !important; }

    /* File uploader */
    [data-testid="stFileUploaderDropzone"] {
        background-color: var(--panel);
        border: 1px dashed var(--line-on-ink);
        border-radius: 6px;
    }

    footer, #MainMenu { visibility: hidden; }
    </style>

    <div class="cw-hero">
        <h1>Cutaway</h1>
        <hr class="cw-stitch" />
        <p>Upload a green-screen product photo and pull a clean, transparent cutout of the garment — ready for a catalogue, a listing, or a spec sheet.</p>
    </div>
    """,
    unsafe_allow_html=True,
)


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


uploaded_file = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is not None:
    original_image = Image.open(uploaded_file).convert("RGB")
    detected_hex = detect_background_hex(np.array(original_image))

    st.sidebar.markdown('<div class="cw-sidebar-title">Chroma key settings</div>', unsafe_allow_html=True)
    st.sidebar.markdown(f'<div class="cw-sidebar-sub">Detected background: {detected_hex}</div>', unsafe_allow_html=True)
    key_color_hex = st.sidebar.color_picker("Background color", detected_hex)
    tola = st.sidebar.slider("Edge softness (tola)", 1, 50, 10, help="Distance below which a pixel is fully kept as foreground.")
    tolb = st.sidebar.slider("Edge softness (tolb)", tola + 1, 120, 60, help="Distance above which a pixel is fully treated as background. Raise this if uneven lighting leaves background pixels partially opaque.")
    erode_px = st.sidebar.slider("Fringe removal strength", 0, 5, 2, help="How many pixels in from the edge are treated as 'trusted' garment color, which then gets grown back outward to replace green-tinted spill pixels. Raise if a green rim persists; lower if fine details look smeared.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<span class="cw-chip">Original</span>', unsafe_allow_html=True)
        st.image(original_image, width="stretch")

    with col2:
        st.markdown('<span class="cw-chip">Extracted</span>', unsafe_allow_html=True)
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
                    label="Save PNG",
                    data=byte_im,
                    file_name="chromakey_cutout.png",
                    mime="image/png",
                    width="stretch",
                )
            except Exception as e:
                st.error(f"Couldn't process this image: {e}")
else:
    st.markdown(
        '<p style="opacity:0.65; padding: 0 0.25rem;">Drop a green-screen photo above to get started.</p>',
        unsafe_allow_html=True,
    )
