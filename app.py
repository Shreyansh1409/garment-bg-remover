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
    page_title="Garment Extractor Pro",
    page_icon="✂️",
    initial_sidebar_state="expanded"
)

if "fringe_val" not in st.session_state:
    st.session_state.fringe_val = 0

def step_fringe(delta: int):
    st.session_state.fringe_val = max(0, min(5, st.session_state.fringe_val + delta))

# ---------------------------------------------------------------------------
# UI Styling
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 2rem !important; max-width: 1400px; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    .app-header { margin-bottom: 2rem; padding-bottom: 1rem; border-bottom: 1px solid rgba(128, 128, 128, 0.2); }
    .app-title { font-weight: 700; font-size: 2.25rem; margin-bottom: 0.25rem; }
    .app-subtitle { font-size: 1rem; opacity: 0.7; }
    .sidebar-header { font-weight: 600; font-size: 1.1rem; margin-top: 1rem; margin-bottom: 0.5rem; }
    .stButton > button { background-color: #4F46E5 !important; color: white !important; border-radius: 8px !important; border: none !important; font-weight: 500 !important; width: 100%; }
    .stButton > button:hover { background-color: #4338CA !important; }
    .stDownloadButton > button { background-color: #10B981 !important; color: white !important; border-radius: 8px !important; border: none !important; font-weight: 600 !important; width: 100%; margin-top: 0.75rem; }
    .stDownloadButton > button:hover { background-color: #059669 !important; }
    [data-testid="stImage"] {
        background-color: #ffffff !important;
        background-image:
            linear-gradient(45deg, #ececec 25%, transparent 25%),
            linear-gradient(135deg, #ececec 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, #ececec 75%),
            linear-gradient(135deg, transparent 75%, #ececec 75%) !important;
        background-size: 16px 16px !important;
        background-position: 0 0, 8px 0, 8px -8px, 0px 8px !important;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    </style>
    """, unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Engine 1: Segformer (Strict Clothing Extraction)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False, max_entries=1)
def load_segformer():
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    return processor, model

@st.cache_data(show_spinner=False, max_entries=2)
def segformer_cutout(image_bytes: bytes, grow_px: int) -> bytes:
    import torch
    import torch.nn as nn
    
    processor, model = load_segformer()
    source_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    inputs = processor(images=source_img, return_tensors="pt")
    with torch.no_grad(): outputs = model(**inputs)
        
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(logits, size=source_img.size[::-1], mode="bilinear", align_corners=False)
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    clothing_labels = [4, 5, 6, 7] # Upper-clothes, Skirt, Pants, Dress
    garment_mask_arr = np.isin(pred_seg, clothing_labels).astype(np.uint8) * 255
    
    if grow_px > 0:
        kernel = np.ones((3, 3), np.uint8)
        garment_mask_arr = cv2.erode(garment_mask_arr, kernel, iterations=grow_px)
        
    garment_mask = Image.fromarray(garment_mask_arr, mode="L")
    cutout = source_img.convert("RGBA")
    cutout.putalpha(garment_mask)
    
    buf = io.BytesIO()
    cutout.save(buf, format="PNG")
    return buf.getvalue()

# ---------------------------------------------------------------------------
# Engine 2: U2Net Cloth (Local Occusion Repair)
# ---------------------------------------------------------------------------
MAX_MASK_EDGE = 1024

@st.cache_resource(show_spinner=False)
def load_ai_session():
    import onnxruntime as ort
    from rembg.sessions import sessions_class
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.intra_op_num_threads = 1
    return sessions_class["u2net_cloth_seg"]("u2net_cloth_seg", opts)

def _mirror_fill(rgb: np.ndarray, garment: np.ndarray, occluded: np.ndarray):
    out = rgb.copy()
    remaining = occluded.copy()
    pieces, count = label(garment | occluded)

    for index in range(1, count + 1):
        piece = pieces == index
        target = piece & occluded
        fabric = piece & garment & ~occluded
        if target.sum() == 0 or fabric.sum() < 500: continue

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

@st.cache_data(show_spinner=False, max_entries=2)
def u2net_cloth_cutout(image_bytes: bytes, repair: bool, grow_px: int) -> bytes:
    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = source.size

    small = source.copy()
    small.thumbnail((MAX_MASK_EDGE, MAX_MASK_EDGE), Image.LANCZOS)
    parts = load_ai_session().predict(small)
    
    upper, lower = (np.array(p.convert("L").resize((width, height), Image.LANCZOS)) > 127 for p in parts[:2])
    garment = upper | lower
    rgb = np.array(source)

    if repair:
        occluded = binary_fill_holes(garment) & ~garment
        if occluded.any():
            rgb, remaining = _mirror_fill(rgb, garment, occluded)
            if remaining.any():
                rgb = cv2.inpaint(rgb, cv2.dilate(remaining.astype(np.uint8) * 255, np.ones((3, 3), np.uint8), 1), 3, cv2.INPAINT_TELEA)
            garment = garment | occluded
            
    garment_mask = (garment * 255).astype(np.uint8)
    if grow_px > 0:
        garment_mask = cv2.erode(garment_mask, np.ones((3, 3), np.uint8), iterations=grow_px)

    result = Image.fromarray(rgb).convert("RGBA")
    result.putalpha(Image.fromarray(garment_mask))
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()

# ---------------------------------------------------------------------------
# Engine 3: Chroma Key
# ---------------------------------------------------------------------------
def detect_background_hex(img_array: np.ndarray, border: int = 10) -> str:
    h, w, _ = img_array.shape
    border = min(border, h // 2, w // 2) or 1
    samples = np.concatenate([
        img_array[:border, :, :].reshape(-1, 3), img_array[-border:, :, :].reshape(-1, 3),
        img_array[:, :border, :].reshape(-1, 3), img_array[:, -border:, :].reshape(-1, 3),
    ])
    return "#{:02X}{:02X}{:02X}".format(*samples.mean(axis=0).astype(np.uint8))

def chroma_cutout(img_array: np.ndarray, key_hex: str, tola: int, tolb: int, grow_px: int) -> Image.Image:
    hex_str = key_hex.lstrip('#')
    target = np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)
    distances = np.linalg.norm(img_array.astype(np.float32) - target, axis=-1)
    alpha_f = np.clip((distances - tola) / max(tolb - tola, 1), 0.0, 1.0)
    
    final_rgb = img_array.copy()
    if grow_px > 0:
        core = cv2.erode((alpha_f > 0.5).astype(np.uint8), np.ones((3, 3), np.uint8), iterations=grow_px)
        if core.any():
            _, indices = distance_transform_edt(1 - core, return_indices=True)
            final_rgb = np.where(core[..., None].astype(bool), final_rgb, final_rgb[indices[0], indices[1]])

    return Image.fromarray(np.dstack([final_rgb, np.uint8(alpha_f * 255)]), "RGBA")

# ---------------------------------------------------------------------------
# App Interface
# ---------------------------------------------------------------------------
st.markdown('<div class="app-header"><div class="app-title">Garment Extractor Pro</div><div class="app-subtitle">Strict Garment Isolation. Designed to automatically delete human subjects.</div></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload product photo to begin", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
if uploaded_file is None:
    st.info("Upload a garment photo above to launch the extraction studio.")
    st.stop()

image_bytes = uploaded_file.getvalue()
original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
img_array = np.array(original_image)

with st.sidebar:
    st.markdown('<div class="sidebar-header">Engine Settings</div>', unsafe_allow_html=True)
    st.error("⚠️ **Garments Only.** This tool explicitly deletes human skin, hair, and body parts.")
    
    method = st.radio("Processing Engine", ["Segformer (Pro Garment AI)", "U2Net (Local Occlusion Repair)", "Chroma Key (Studio Green)", "Hybrid (AI + Chroma Key)"], label_visibility="collapsed")
    st.markdown("---")
    
    if method == "U2Net (Local Occlusion Repair)":
        st.info("Maps generic clothing shapes and attempts to mathematically rebuild fabric hidden behind arms.")
        repair_occlusion = st.checkbox("Rebuild fabric behind arms", value=True)
        
    elif method == "Chroma Key (Studio Green)":
        st.warning("Math-based background subtraction for solid colors.")
        detected_hex = detect_background_hex(img_array)
        key_color_hex = st.color_picker("Key Color", detected_hex)
        tola = st.slider("Shadow Tolerance", 1, 50, 10)
        tolb = st.slider("Highlight Tolerance", tola + 1, 120, 60)
        
    elif method == "Hybrid (AI + Chroma Key)":
        st.info("⚡ **Combines U2Net AI with precise Chroma Keying.** Punches out enclosed background gaps and refines edges.")
        repair_occlusion = st.checkbox("Rebuild fabric behind arms", value=False)
        detected_hex = detect_background_hex(img_array)
        key_color_hex = st.color_picker("Key Color", detected_hex)
        tola = st.slider("Shadow Tolerance", 1, 50, 10)
        tolb = st.slider("Highlight Tolerance", tola + 1, 120, 60)
        
    with st.expander("Edge Cleanup", expanded=True):
        col_min, col_slide, col_plus = st.columns([1, 4, 1])
        with col_min: st.button("−", on_click=step_fringe, args=(-1,))
        with col_slide: grow_px = st.slider("Fringe", 0, 5, key="fringe_val", label_visibility="collapsed")
        with col_plus: st.button("+", on_click=step_fringe, args=(1,))

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="image-card-title">Source Image</div>', unsafe_allow_html=True)
    st.image(original_image, use_container_width=True)

with col2:
    st.markdown('<div class="image-card-title">Extraction Result</div>', unsafe_allow_html=True)
    
    try:
        with st.spinner("Extracting garments..."):
            if method == "Segformer (Pro Garment AI)":
                out_bytes = segformer_cutout(image_bytes, grow_px)
                extracted_image = Image.open(io.BytesIO(out_bytes))
            elif method == "U2Net (Local Occlusion Repair)":
                out_bytes = u2net_cloth_cutout(image_bytes, repair_occlusion, grow_px)
                extracted_image = Image.open(io.BytesIO(out_bytes))
            elif method == "Chroma Key (Studio Green)":
                extracted_image = chroma_cutout(img_array, key_color_hex, tola, tolb, grow_px)
            elif method == "Hybrid (AI + Chroma Key)":
                ai_bytes = u2net_cloth_cutout(image_bytes, repair_occlusion, 0)
                ai_alpha = np.array(Image.open(io.BytesIO(ai_bytes)).split()[-1])
                chroma_img = chroma_cutout(img_array, key_color_hex, tola, tolb, grow_px)
                chroma_alpha = np.array(chroma_img.split()[-1])
                combined_alpha = np.minimum(ai_alpha, chroma_alpha)
                extracted_image = chroma_img.copy()
                extracted_image.putalpha(Image.fromarray(combined_alpha))
                
        st.image(extracted_image, use_container_width=True)
        
        buf = io.BytesIO()
        extracted_image.save(buf, format="PNG")
        st.download_button("↓ Export Transparent Garment", data=buf.getvalue(), file_name="garment_asset.png", mime="image/png")
            
    except Exception as exc: 
        st.error(f"Processing Error: {exc}")
