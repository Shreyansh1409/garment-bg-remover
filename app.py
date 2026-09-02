import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label

# ---------------------------------------------------------------------------
# App Configuration & Session State
# ---------------------------------------------------------------------------
st.set_page_config(
    layout="wide",
    page_title="TexAI Extractor Pro",
    page_icon="✨",
    initial_sidebar_state="expanded",
)

if "fringe_val" not in st.session_state:
    st.session_state.fringe_val = 0


def step_fringe(delta: int):
    st.session_state.fringe_val = max(0, min(5, st.session_state.fringe_val + delta))


# ---------------------------------------------------------------------------
# Modern App CSS Styling (Dark Studio Theme)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp { background-color: #0b0f19; }

    .block-container { padding-top: 2rem !important; max-width: 1400px; }
    header { visibility: hidden; }
    footer { visibility: hidden; }

    .app-header {
        margin-bottom: 2.5rem;
        padding-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.08);
    }
    .app-title {
        font-weight: 700;
        font-size: 2.8rem;
        letter-spacing: -0.02em;
        margin-bottom: 0.25rem;
        background: linear-gradient(90deg, #ffffff 0%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .app-subtitle { font-size: 1.05rem; color: #9ca3af; font-weight: 400; }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    .sidebar-header {
        font-weight: 600;
        font-size: 0.85rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin: 1.5rem 0 0.75rem 0;
    }

    .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #4338ca 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(79, 70, 229, 0.2) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(79, 70, 229, 0.4) !important;
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.8rem 1.2rem !important;
        margin-top: 1rem;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 12px rgba(16, 185, 129, 0.2) !important;
    }
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(16, 185, 129, 0.4) !important;
    }

    .image-card-title {
        font-weight: 600;
        font-size: 0.85rem;
        color: #9ca3af;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    [data-testid="stFileUploaderDropzone"] {
        border-radius: 16px;
        border: 2px dashed rgba(255, 255, 255, 0.15);
        background-color: rgba(255, 255, 255, 0.02);
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #4f46e5;
        background-color: rgba(79, 70, 229, 0.05);
    }

    [data-testid="stImage"] {
        background-color: #ffffff !important;
        background-image:
            linear-gradient(45deg, #ececec 25%, transparent 25%),
            linear-gradient(135deg, #ececec 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, #ececec 75%),
            linear-gradient(135deg, transparent 75%, #ececec 75%) !important;
        background-size: 16px 16px !important;
        background-position: 0 0, 8px 0, 8px -8px, 0px 8px !important;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.05);
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.2);
        overflow: hidden;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# ONNX engines
#
# Segformer was removed. It brought torch + transformers, and the deploy log
# showed exactly what that costs: dependencies installed fine and uvicorn came
# up, then "Loading weights: 100% 380/380" was followed by the rembg model
# download and then silence — no Python traceback, process gone. That is the
# signature of an OOM kill, which cannot be caught or logged. torch alone is
# 583 MB resident on import, before Streamlit's ~230 MB and before either
# model's weights.
#
# (The same log also shows an outbound Hugging Face request succeeding, so the
# "Streamlit blocks Hugging Face by DNS" theory was never right.)
#
# Steady-state RSS measured on a real 1037x1716 photo, over a ~230 MB baseline,
# against Streamlit Community Cloud's 690 MB floor / 2.7 GB ceiling:
#
#   u2netp           317 MB   whole subject, works on flat-lays AND on-model
#   u2net_cloth_seg  665 MB   clothes only, needs a body in frame
#
# Only ONE is ever resident: cache_resource(max_entries=1) evicts the previous
# session when the model changes, which is what keeps a mode switch from
# stacking both.
# ---------------------------------------------------------------------------
SUBJECT_MODEL = "u2netp"
CLOTH_MODEL = "u2net_cloth_seg"
MAX_MASK_EDGE = 1024


@st.cache_resource(show_spinner=False, max_entries=1)
def load_ai_session(model_name: str):
    """Memory-constrained ONNX session. rembg's new_session() builds its own
    SessionOptions with no way to pass ours, so the class is built directly.

    enable_cpu_mem_arena=False matters more than it looks: with the arena on,
    RSS climbed 531 -> 731 MB between the first and second inference and never
    came back down. Off, it stays flat, and it is marginally faster here.
    """
    import onnxruntime as ort
    from rembg.sessions import sessions_class

    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1

    return {cls.name(): cls for cls in sessions_class}[model_name](model_name, opts)


def _shrink(source: Image.Image) -> Image.Image:
    small = source.copy()
    small.thumbnail((MAX_MASK_EDGE, MAX_MASK_EDGE), Image.LANCZOS)
    return small


SPILL_PX = 2  # width of the contaminated edge band to repaint


def _decontaminate(rgb: np.ndarray, mask: np.ndarray, px: int = SPILL_PX) -> np.ndarray:
    """Repaint the edge band with real garment colour taken from just inside it.

    Every partially transparent edge pixel in the source photo is a BLEND of
    garment and backdrop, so it carries the backdrop's colour. On a green screen
    that is a visible green halo: measured on a real flat-lay, 69.6% of edge
    pixels were green-tinted, mean RGB (60,106,56).

    Unpremultiplying against the key colour is the textbook fix and only got
    that to 44.9%, because a segmentation mask is a probability map, not a true
    matte, so the alpha in the equation is wrong. Eroding to a trusted core and
    growing its colour outward does not depend on alpha at all and reaches 6.4%.
    The ALPHA IS LEFT UNTOUCHED, so the silhouette and its antialiasing survive
    — only the colour under the edge changes.
    """
    core = cv2.erode((mask > 127).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=px)
    if not core.any():
        return rgb
    _, indices = distance_transform_edt(1 - core, return_indices=True)
    return np.where(core[..., None].astype(bool), rgb, rgb[indices[0], indices[1]])


def _finish(rgb: np.ndarray, mask: np.ndarray, grow_px: int) -> bytes:
    """Decontaminate the edge, apply one — and only one — erosion, then encode."""
    rgb = _decontaminate(rgb, mask)
    if grow_px > 0:
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=grow_px)
    result = Image.fromarray(rgb).convert("RGBA")
    result.putalpha(Image.fromarray(mask))
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Engine 1: whole subject (default)
#
# The only engine that works on a flat-lay. Both clothing models need a body to
# anchor the upper-body region: on a Gemini flat-lay, u2net_cloth_seg returned
# the trousers at 18.6% of frame but the corset at 0.2% — shredded. u2netp on
# the same photo scored 99% of the garment with 0% background left behind.
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, max_entries=2)
def subject_cutout(image_bytes: bytes, grow_px: int) -> bytes:
    from rembg import remove

    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask = remove(_shrink(source), session=load_ai_session(SUBJECT_MODEL), only_mask=True)
    mask = np.array(mask.resize(source.size, Image.LANCZOS))
    return _finish(np.array(source), mask, grow_px)


# ---------------------------------------------------------------------------
# Engine 2: clothes only, with occlusion repair
# ---------------------------------------------------------------------------
def _mirror_fill(rgb: np.ndarray, garment: np.ndarray, occluded: np.ndarray):
    """Fill hidden fabric from the mirrored side of the same garment piece.

    Garments are near bilaterally symmetric, so fabric behind an arm usually has
    a real, photographed counterpart. Copying it invents nothing. Ground-truth
    mean absolute error on a synthetic arm occlusion: 46.7 untouched, 28.0 for
    cv2.inpaint, 24.1 for this. The symmetry axis is the bounding-box midpoint,
    not the centroid — the centroid is dragged sideways by whatever the arm
    removed and scored 28.3, giving up the whole advantage.
    """
    out = rgb.copy()
    remaining = occluded.copy()
    pieces, count = label(garment | occluded)

    for index in range(1, count + 1):
        piece = pieces == index
        target = piece & occluded
        fabric = piece & garment & ~occluded
        if target.sum() == 0 or fabric.sum() < 500:
            continue

        xs = np.nonzero(fabric)[1]
        axis = int(round((xs.min() + xs.max()) / 2))

        ys, xs_t = np.nonzero(target)
        mirrored = 2 * axis - xs_t
        inside = (mirrored >= 0) & (mirrored < rgb.shape[1])
        ys, xs_t, mirrored = ys[inside], xs_t[inside], mirrored[inside]

        usable = fabric[ys, mirrored]
        out[ys[usable], xs_t[usable]] = rgb[ys[usable], mirrored[usable]]
        remaining[ys[usable], xs_t[usable]] = False

    return out, remaining


def _garment_mask_and_rgb(image_bytes: bytes, repair: bool):
    """Shared by the garment and hybrid paths so the model runs once."""
    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = source.size

    # predict() returns the three masks (upper, lower, full) as a list.
    # remove(only_mask=True) instead returns ONE image at 3x the height with
    # them stacked; squashing that back gives a ~6% alpha smear that reads as a
    # model failure but is a plumbing bug. remove() also swallows unknown
    # keywords via **kwargs, so an invented return_multiple=True is a no-op.
    parts = load_ai_session(CLOTH_MODEL).predict(_shrink(source))
    upper, lower = (
        np.array(p.convert("L").resize((width, height), Image.LANCZOS)) > 127
        for p in parts[:2]
    )
    garment = upper | lower
    rgb = np.array(source)

    if repair:
        # ENCLOSED gaps only. An arm across the torso encloses a hole; the bare
        # midriff between a crop top and trousers does not. A morphological
        # closing would bridge that midriff and weld a two-piece into a
        # jumpsuit, so it is deliberately not used here.
        occluded = binary_fill_holes(garment) & ~garment
        if occluded.any():
            rgb, remaining = _mirror_fill(rgb, garment, occluded)
            if remaining.any():
                rgb = cv2.inpaint(
                    rgb,
                    cv2.dilate(remaining.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), 1),
                    3,
                    cv2.INPAINT_TELEA,
                )
            garment = garment | occluded

    return rgb, (garment * 255).astype(np.uint8)


@st.cache_data(show_spinner=False, max_entries=2)
def garment_cutout(image_bytes: bytes, repair: bool, grow_px: int) -> bytes:
    rgb, mask = _garment_mask_and_rgb(image_bytes, repair)
    return _finish(rgb, mask, grow_px)


# ---------------------------------------------------------------------------
# Engine 3: chroma key
#
# RGB distance, not Cb/Cr — including luma is what lets it keep a black or
# cream garment (0% -> 93% and 8% -> 97% on the two test photos). The cost is
# that a lit or textured backdrop also reads as "not the key colour" and
# survives: on a room interior it kept 100% of the corner. Genuinely good on a
# flat, evenly lit sweep; misleading anywhere else.
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
    mid = np.median(samples, axis=0).astype(np.uint8)
    return "#{:02X}{:02X}{:02X}".format(*mid)


def chroma_alpha(img_array: np.ndarray, key_hex: str, tola: int, tolb: int) -> np.ndarray:
    hex_str = key_hex.lstrip("#")
    target = np.array([int(hex_str[i:i + 2], 16) for i in (0, 2, 4)], dtype=np.float32)
    distances = np.linalg.norm(img_array.astype(np.float32) - target, axis=-1)
    return np.clip((distances - tola) / max(tolb - tola, 1), 0.0, 1.0)


def chroma_cutout(img_array: np.ndarray, key_hex: str, tola: int, tolb: int, grow_px: int) -> bytes:
    alpha_f = chroma_alpha(img_array, key_hex, tola, tolb)
    return _finish(img_array.copy(), np.uint8(alpha_f * 255), grow_px)


# ---------------------------------------------------------------------------
# App Interface
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-header"><div class="app-title">TexAI Extractor</div>'
    '<div class="app-subtitle">High-fidelity garment isolation. Clean catalogue '
    'assets from flat-lays or on-model shots.</div></div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Upload product photo to begin", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
)

if uploaded_file is None:
    st.info("Upload a garment photo above to launch the extraction studio.")
    st.stop()

image_bytes = uploaded_file.getvalue()
original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
img_array = np.array(original_image)

with st.sidebar:
    st.markdown('<div class="sidebar-header">Extraction Engine</div>', unsafe_allow_html=True)

    method = st.radio(
        "Processing Engine",
        [
            "Subject cutout (any photo)",
            "Garment only (needs a person)",
            "Chroma key (solid backdrop)",
            "Hybrid (garment + chroma)",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    if method == "Subject cutout (any photo)":
        st.info(
            "Keeps everything that isn't background — a flat-lay, or a model still "
            "wearing the garment. The only engine that works on a flat-lay, and the "
            "lightest at ~317 MB."
        )

    elif method == "Garment only (needs a person)":
        st.warning(
            "Deletes skin, hair and shoes, keeping just the clothes. Needs a person "
            "in frame — on a flat-lay it shreds the upper garment. ~665 MB."
        )
        repair_occlusion = st.checkbox(
            "Rebuild fabric behind arms",
            value=True,
            help="Fills only fully enclosed gaps, by mirroring the same piece's other "
                 "side. A bare midriff is left alone.",
        )

    elif method == "Chroma key (solid backdrop)":
        st.warning(
            "Colour-distance subtraction. Reliable on a flat, evenly lit sweep; on a "
            "room or a gradient backdrop it keeps the background."
        )
        detected_hex = detect_background_hex(img_array)
        key_color_hex = st.color_picker("Key Color", detected_hex)
        tola = st.slider("Shadow Tolerance", 1, 50, 40)
        tolb = st.slider("Highlight Tolerance", tola + 1, 120, 60)

    else:
        st.info(
            "Intersects the garment mask with the chroma mask. It can only ever "
            "remove pixels, never restore them, so the result is bounded by "
            "whichever of the two is worse."
        )
        repair_occlusion = st.checkbox("Rebuild fabric behind arms", value=False)
        detected_hex = detect_background_hex(img_array)
        key_color_hex = st.color_picker("Key Color", detected_hex)
        tola = st.slider("Shadow Tolerance", 1, 50, 40)
        tolb = st.slider("Highlight Tolerance", tola + 1, 120, 60)

    with st.expander("Edge Cleanup", expanded=True):
        col_min, col_slide, col_plus = st.columns([1, 4, 1])
        with col_min:
            st.button("−", on_click=step_fringe, args=(-1,), width="stretch")
        with col_slide:
            grow_px = st.slider("Fringe", 0, 5, key="fringe_val", label_visibility="collapsed")
        with col_plus:
            st.button("+", on_click=step_fringe, args=(1,), width="stretch")

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="image-card-title">Source Image</div>', unsafe_allow_html=True)
    st.image(original_image, width="stretch")

with col2:
    st.markdown('<div class="image-card-title">Extraction Result</div>', unsafe_allow_html=True)

    try:
        with st.spinner("Extracting… the first run for each engine downloads its model."):
            if method == "Subject cutout (any photo)":
                out_bytes = subject_cutout(image_bytes, grow_px)

            elif method == "Garment only (needs a person)":
                out_bytes = garment_cutout(image_bytes, repair_occlusion, grow_px)

            elif method == "Chroma key (solid backdrop)":
                out_bytes = chroma_cutout(img_array, key_color_hex, tola, tolb, grow_px)

            else:
                # Erode ONCE, at the end. Eroding inside each branch and again on
                # the combined mask is a triple pass: at fringe=5 that took an
                # 8 px strap to zero and ate 27% of the garment area.
                rgb, garment_mask = _garment_mask_and_rgb(image_bytes, repair_occlusion)
                chroma_mask = np.uint8(chroma_alpha(img_array, key_color_hex, tola, tolb) * 255)
                out_bytes = _finish(rgb, np.minimum(garment_mask, chroma_mask), grow_px)

        extracted_image = Image.open(io.BytesIO(out_bytes))
        st.image(extracted_image, width="stretch")
        st.download_button(
            "↓ Export Transparent Garment",
            data=out_bytes,
            file_name="garment_asset.png",
            mime="image/png",
            width="stretch",
        )

    except ModuleNotFoundError as exc:
        st.error(f"Missing dependency: `{exc.name}`")
        st.info("Add `rembg` and `onnxruntime` to requirements.txt, then reboot the app.")
    except Exception as exc:  # noqa: BLE001
        st.error(f"Processing Error: {exc}")
