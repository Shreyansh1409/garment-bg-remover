import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from scipy.ndimage import distance_transform_edt

st.set_page_config(layout="wide", page_title="Cutaway — Garment Extractor", page_icon="✂️")

# ---------------------------------------------------------------------------
# Styling
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

    [data-testid="stSlider"] [role="slider"] { background-color: var(--thread) !important; }
    [data-testid="stSlider"] div[style*="background-color: rgb(255, 75, 75)"] { background-color: var(--thread) !important; }

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
        <p>Pull a clean, transparent cutout of a garment — ready for a catalogue, a listing, or a spec sheet.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# AI cutout (segmentation / matting)
# ---------------------------------------------------------------------------
AI_MODELS = {
    "u2net_cloth_seg — garments only (removes skin)": "u2net_cloth_seg",
    "u2netp — light and fast (general cutout)": "u2netp",
    "u2net — slightly cleaner, ~2x memory (general cutout)": "u2net",
}

MAX_MASK_EDGE = 1024  # longest edge used for mask inference


@st.cache_resource(show_spinner=False)
def load_ai_session(model_name: str):
    import onnxruntime as ort
    from rembg.sessions import sessions_class

    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1

    session_cls = {cls.name(): cls for cls in sessions_class}[model_name]
    return session_cls(model_name, opts)


@st.cache_data(show_spinner=False, max_entries=4)
def ai_cutout(image_bytes: bytes, model_name: str) -> bytes:
    from rembg import remove

    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    small = source.copy()
    small.thumbnail((MAX_MASK_EDGE, MAX_MASK_EDGE), Image.LANCZOS)
    
    raw_mask = remove(small, session=load_ai_session(model_name), only_mask=True)
    
    # Robust extraction: catches edge cases where the rembg API bypasses only_mask=True 
    # and returns a 4-channel RGBA cutout instead of a 1-channel L mask. 
    channels = raw_mask.split()
    mask = channels[-1] if len(channels) == 4 else raw_mask.convert("L")

    result = source.convert("RGBA")
    result.putalpha(mask.resize(source.size, Image.LANCZOS))

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Chroma key
# ---------------------------------------------------------------------------
def detect_background_hex(img_array: np.ndarray, border: int = 10) -> str:
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


def chroma_cutout(img_array: np.ndarray, key_hex: str, tola: int, tolb: int, grow_px: int) -> Image.Image:
    import chromakey
    out, mask = chromakey.chroma_key(img_array, key_hex, tola=tola, tolb=tolb)
    alpha_f = 1 - mask

    final_rgb = out
    if grow_px > 0:
        fg_binary = (alpha_f > 0.5).astype(np.uint8)
        core = cv2.erode(fg_binary, np.ones((3, 3), np.uint8), iterations=grow_px)
        if core.any():
            _, indices = distance_transform_edt(1 - core, return_indices=True)
            grown_rgb = out[indices[0], indices[1]]
            final_rgb = np.where(core[..., None].astype(bool), out, grown_rgb)

    alpha = np.uint8(np.clip(alpha_f, 0, 1) * 255)
    rgba = np.dstack([final_rgb, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a photo", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
)

if uploaded_file is None:
    st.markdown(
        '<p style="opacity:0.65; padding: 0 0.25rem;">Drop a garment photo above to get started.</p>',
        unsafe_allow_html=True,
    )
    st.stop()

image_bytes = uploaded_file.getvalue()
original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
img_array = np.array(original_image)

st.sidebar.markdown('<div class="cw-sidebar-title">Extraction</div>', unsafe_allow_html=True)
st.sidebar.markdown(
    '<div class="cw-sidebar-sub">AI cutout reads the garment\'s shape. '
    'Chroma key reads one background colour — only use it on a real green screen.</div>',
    unsafe_allow_html=True,
)
method = st.sidebar.radio(
    "Method",
    ["AI cutout", "Chroma key"],
    label_visibility="collapsed"
)

if method == "AI cutout":
    model_label = st.sidebar.selectbox("Model", list(AI_MODELS), index=0)
    model_name = AI_MODELS[model_label]
    
    st.sidebar.markdown(
        '<div class="cw-sidebar-sub" style="margin-top:0.6rem;">Runs single-threaded '
        'with the ONNX memory arena off, and masks at 1024&nbsp;px, to hold steady '
        'memory near 320&nbsp;MB. The download is still full resolution.</div>',
        unsafe_allow_html=True,
    )
else:
    detected_hex = detect_background_hex(img_array)
    st.sidebar.markdown(
        f'<div class="cw-sidebar-sub">Detected background: {detected_hex}</div>',
        unsafe_allow_html=True,
    )
    key_color_hex = st.sidebar.color_picker("Background color", detected_hex)
    tola = st.sidebar.slider("Solid background below (tola)", 1, 50, 10)
    tolb = st.sidebar.slider("Solid garment above (tolb)", tola + 1, 120, 60)
    grow_px = st.sidebar.slider("Fringe removal strength", 0, 5, 2)

col1, col2 = st.columns(2)

with col1:
    st.markdown('<span class="cw-chip">Original</span>', unsafe_allow_html=True)
    st.image(original_image, width="stretch")

with col2:
    st.markdown('<span class="cw-chip">Extracted</span>', unsafe_allow_html=True)
    try:
        if method == "AI cutout":
            with st.spinner("Cutting out the garment… the first run downloads the model."):
                extracted_image = Image.open(io.BytesIO(ai_cutout(image_bytes, model_name)))
            note = model_label.split(" — ")[0]
        else:
            with st.spinner("Calculating chroma key…"):
                extracted_image = chroma_cutout(img_array, key_color_hex, tola, tolb, grow_px)
            note = f"chroma key on {key_color_hex}"

        st.image(extracted_image, width="stretch")
        st.caption(note)

        buf = io.BytesIO()
        extracted_image.save(buf, format="PNG")
        st.download_button(
            label="Save PNG",
            data=buf.getvalue(),
            file_name="cutout.png",
            mime="image/png",
            width="stretch",
        )
    except ModuleNotFoundError as exc:
        st.error(
            f"Missing dependency: {exc.name}. Add `rembg` and `onnxruntime` to "
            "requirements.txt for the AI path, or switch to Chroma key."
        )
    except Exception as exc: 
        st.error(f"Couldn't process this image: {exc}")
