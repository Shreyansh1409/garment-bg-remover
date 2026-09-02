import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image, ImageDraw
from scipy.ndimage import binary_fill_holes, distance_transform_edt, label

# Lasso selection uses plotly through st.plotly_chart(on_select=...), which is a
# FIRST-PARTY Streamlit API. The previous attempt used streamlit-drawable-canvas
# and failed three times running: its latest release still calls
# streamlit.elements.image.image_to_url(), a private helper Streamlit deleted, and
# each shim only exposed the next incompatibility. Its frontend is built against
# an old component protocol and cannot be verified without a browser. plotly costs
# 62 MB of build but 0 MB of RSS, because it imports lazily.
try:
    import plotly.graph_objects as go

    LASSO_AVAILABLE = True
except Exception:  # noqa: BLE001
    go = None
    LASSO_AVAILABLE = False

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

    distance, indices = distance_transform_edt(1 - core, return_indices=True)

    # Repaint ONLY a narrow band outside the core. The nearest-core lookup is
    # defined everywhere, so applying it to the whole frame fills the entire
    # transparent background with rays of garment colour radiating from the
    # silhouette. That is invisible while alpha is respected — and lands as a
    # streaked mess the moment anything downstream flattens the PNG or drops the
    # alpha channel, which is how these files reach a try-on API or a CMS.
    # Every VISIBLE pixel outside the core, whatever the halo's width — a fixed
    # band was tried at px+2 and let edge-green back up from 6.7% to 15.6%,
    # because the kept-backdrop halo runs to ~9px on some photos.
    band = (distance > 0) & (mask > 0)
    repainted = rgb.copy()
    repainted[band] = rgb[indices[0][band], indices[1][band]]
    return repainted


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
def _subject_mask_and_rgb(image_bytes: bytes):
    """Shared by the subject and hybrid paths so the model runs once."""
    from rembg import remove

    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    mask = remove(_shrink(source), session=load_ai_session(SUBJECT_MODEL), only_mask=True)
    return np.array(source), np.array(mask.resize(source.size, Image.LANCZOS))


@st.cache_data(show_spinner=False, max_entries=2)
def subject_cutout(image_bytes: bytes, grow_px: int) -> bytes:
    rgb, mask = _subject_mask_and_rgb(image_bytes)
    return _finish(rgb, mask, grow_px)


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
# Selective cleanup
#
# For residue no global step can catch. A drop shadow is the case that matters:
# it is not the backdrop colour, so the chroma mask never sees it, and it is
# attached to the garment, so the AI mask keeps it. Nothing automatic in this
# app removes it.
#
# A shadow IS, however, the backdrop at lower brightness. So the first test is
# chromaticity, not colour: normalise a pixel by its own total brightness and
# compare that to the backdrop's normalised value. Same hue + darker = shadow.
#
# Chromaticity ALONE is not enough, and assuming otherwise destroys garments.
# Near-black normalises to neutral (0.33,0.33,0.33) — which is exactly a grey
# backdrop's chromaticity. Measured on a grey studio flat-lay, the hue test on
# its own took the black trousers from 87.4% surviving to 6.0% at the mildest
# setting. It looked safe on a green screen only because green is saturated.
#
# BRIGHTNESS_FLOOR is what makes it safe. A shadow keeps a real fraction of the
# backdrop's luminance; black satin keeps almost none (ratio ~0.15 against a
# grey sweep). Cutting only pixels between the floor and full backdrop
# brightness separates the two on neutral backdrops as well as saturated ones.
#
# Applied to the finished PNG rather than inside the cached engines, so dragging
# these sliders re-runs a few array ops instead of the model.
# ---------------------------------------------------------------------------
# Swept against both failure cases on real photos. Below 0.30 the garment itself
# starts going (88.5% -> 83.6% -> 66.0% at 0.25); at 0.45 a real green shadow
# measured at ratio 0.44 sat just under the gate and survived. 0.35 clears more
# shadow while black satin on a grey sweep stays at 87.4%, unchanged.
BRIGHTNESS_FLOOR = 0.35


def box_region(shape, box_pct) -> np.ndarray:
    """Rectangle selection, as a boolean mask at image resolution."""
    height, width = shape
    (x0, x1), (y0, y1) = box_pct
    region = np.zeros((height, width), bool)
    region[
        int(height * y0 / 100):max(int(height * y1 / 100), int(height * y0 / 100) + 1),
        int(width * x0 / 100):max(int(width * x1 / 100), int(width * x0 / 100) + 1),
    ] = True
    return region


def lasso_region(selection, shape, grid_w: int, grid_h: int) -> np.ndarray:
    """Turn a lasso/box selection over the point grid into a full-resolution mask.

    An invisible grid of points is drawn over the result; plotly returns the ones
    inside the lasso. Points are emitted row-major, so a point's flat index gives
    back its cell, and the low-resolution cell mask is then scaled up.
    """
    points = (selection or {}).get("points", []) or []
    if not points:
        return np.zeros(shape, bool)

    cells = np.zeros((grid_h, grid_w), bool)
    for point in points:
        index = point.get("point_index", point.get("pointIndex"))
        if index is None or index >= grid_w * grid_h:
            continue
        cells[index // grid_w, index % grid_w] = True

    if not cells.any():
        return np.zeros(shape, bool)
    return cv2.resize(cells.astype(np.uint8), (shape[1], shape[0]),
                      interpolation=cv2.INTER_NEAREST) > 0


def selective_cleanup(png_bytes: bytes, key_hex: str, region: np.ndarray, shadow: int, trim: int) -> bytes:
    """Trim shadow and border residue inside the selected region only."""
    if (shadow <= 0 and trim <= 0) or region is None or not region.any():
        return png_bytes

    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    arr = np.array(img)
    rgb = arr[:, :, :3].astype(np.float32)
    alpha = arr[:, :, 3]

    if shadow > 0:
        key = np.array([int(key_hex.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
        total = rgb.sum(axis=-1, keepdims=True) + 1e-6
        chromaticity = rgb / total
        key_chroma = key / (key.sum() + 1e-6)
        hue_gap = np.linalg.norm(chromaticity - key_chroma, axis=-1) * 255
        ratio = rgb.sum(-1) / (key.sum() + 1e-6)
        shadowed = (hue_gap < shadow) & (ratio < 1.0) & (ratio > BRIGHTNESS_FLOOR)
        alpha = np.where(region & shadowed, 0, alpha)

    if trim > 0:
        trimmed = cv2.erode(alpha, np.ones((3, 3), np.uint8), iterations=trim)
        alpha = np.where(region, trimmed, alpha)

    out = Image.fromarray(np.dstack([arr[:, :, :3], alpha]).astype(np.uint8), "RGBA")
    buf = io.BytesIO()
    out.save(buf, format="PNG")
    return buf.getvalue()


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
            "Subject cutout, then anything still matching the key colour is "
            "subtracted — for shadows or backdrop the AI kept. It can only remove "
            "pixels, never restore them, so reach for it only when Subject cutout "
            "leaves something behind."
        )
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

        st.divider()
        selective = st.checkbox(
            "Selective cleanup",
            value=False,
            help="Clean a chosen area only. For a drop shadow, which is not the "
                 "backdrop colour and so survives every other step in this app.",
        )
        draw_mode = False
        if selective:
            if LASSO_AVAILABLE:
                draw_mode = st.radio(
                    "Selection",
                    ["Lasso on the result", "Rectangle sliders"],
                    label_visibility="collapsed",
                ) == "Lasso on the result"
            else:
                st.caption("Lasso selection needs `plotly` in requirements.txt.")

            if draw_mode:
                lasso_detail = st.select_slider(
                    "Selection detail", [40, 60, 90, 130], value=90,
                    help="Grid resolution behind the lasso. Higher follows the "
                         "outline more closely and is slower to draw.",
                )
            else:
                box_x = st.slider("Region \u2014 left / right %", 0, 100, (0, 100))
                box_y = st.slider("Region \u2014 top / bottom %", 0, 100, (60, 100))

            shadow_strength = st.slider(
                "Shadow removal", 0, 60, 25,
                help="Cuts pixels inside the region that share the backdrop's hue but "
                     "are darker. Black fabric is unaffected \u2014 it is dark but neutral, "
                     "not a dark version of the backdrop. Raise until the shadow goes; "
                     "if real fabric starts disappearing, you have gone too far.",
            )
            extra_trim = st.slider(
                "Extra trim (px)", 0, 6, 0,
                help="Blunt erosion inside the region only, for border residue that "
                     "isn't shadow.",
            )

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="image-card-title">Source Image</div>', unsafe_allow_html=True)
    preview = original_image
    if selective and not draw_mode:
        # Show the rectangle, otherwise the sliders are guesswork.
        preview = original_image.copy()
        w, h = preview.size
        ImageDraw.Draw(preview).rectangle(
            [int(w * box_x[0] / 100), int(h * box_y[0] / 100),
             int(w * box_x[1] / 100) - 1, int(h * box_y[1] / 100) - 1],
            outline=(79, 70, 229), width=max(2, w // 250),
        )
    st.image(preview, width="stretch")

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
                # Intersect the SUBJECT mask, not the cloth mask. Measured against
                # the reference silhouette on a green-screen flat-lay:
                #   cloth_seg alone          6.7% of the garment punched away
                #   min(cloth, chroma)       6.8%   <- what this used to be
                #   min(subject, chroma)     0.8% holes, 0.0% background kept
                # min() can only subtract, so intersecting with the engine that
                # shreds flat-lays guaranteed a shredded result.
                rgb, subject_mask = _subject_mask_and_rgb(image_bytes)
                chroma_mask = np.uint8(chroma_alpha(img_array, key_color_hex, tola, tolb) * 255)
                out_bytes = _finish(rgb, np.minimum(subject_mask, chroma_mask), grow_px)

        if selective:
            raw = Image.open(io.BytesIO(out_bytes)).convert("RGBA")
            shape = (raw.size[1], raw.size[0])

            if draw_mode:
                # Select ON the result, not the source — you can only see what
                # needs erasing once the background is already gone.
                st.caption("Lasso around what should be erased, then it is applied below.")
                grid_w = int(lasso_detail)
                grid_h = max(int(grid_w * raw.size[1] / raw.size[0]), 1)

                # Invisible, selectable grid over the image. Row-major order is
                # what lets lasso_region() map a point index back to a cell.
                ys_grid, xs_grid = np.mgrid[0:grid_h, 0:grid_w]
                figure = go.Figure(
                    go.Scattergl(
                        x=((xs_grid.ravel() + 0.5) * raw.size[0] / grid_w),
                        y=((ys_grid.ravel() + 0.5) * raw.size[1] / grid_h),
                        mode="markers",
                        marker=dict(size=6, opacity=0),
                        hoverinfo="skip",
                        showlegend=False,
                    )
                )
                backdrop = Image.new("RGBA", raw.size, (120, 120, 120, 255))
                figure.add_layout_image(
                    dict(
                        source=Image.alpha_composite(backdrop, raw).convert("RGB"),
                        xref="x", yref="y", x=0, y=0,
                        sizex=raw.size[0], sizey=raw.size[1],
                        sizing="stretch", layer="below",
                    )
                )
                figure.update_xaxes(visible=False, range=[0, raw.size[0]])
                figure.update_yaxes(
                    visible=False, range=[raw.size[1], 0],
                    scaleanchor="x", scaleratio=1,
                )
                figure.update_layout(
                    dragmode="lasso",
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=460,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                event = st.plotly_chart(
                    figure,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode=("lasso", "box"),
                    key="cleanup_lasso",
                )
                region = lasso_region(
                    (event or {}).get("selection"), shape, grid_w, grid_h
                )
            else:
                region = box_region(shape, (box_x, box_y))

            out_bytes = selective_cleanup(
                out_bytes,
                detect_background_hex(img_array),
                region,
                shadow_strength,
                extra_trim,
            )

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
