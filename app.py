import io
import base64
import requests

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# App Configuration & Callbacks
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Garment Extractor Pro", page_icon="✂️", initial_sidebar_state="expanded")

if "fringe_val" not in st.session_state:
    st.session_state.fringe_val = 0

def step_fringe(delta):
    st.session_state.fringe_val = max(0, min(5, st.session_state.fringe_val + delta))

# ---------------------------------------------------------------------------
# Modern App CSS
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .block-container { padding-top: 2rem !important; max-width: 1400px; }
    header { visibility: hidden; }
    footer { visibility: hidden; }

    .app-header {
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid rgba(128, 128, 128, 0.2);
    }
    .app-title { font-weight: 700; font-size: 2.25rem; margin-bottom: 0.25rem; }
    .app-subtitle { font-size: 1rem; opacity: 0.7; }
    .sidebar-header { font-weight: 600; font-size: 1.1rem; margin-top: 1rem; margin-bottom: 0.5rem; }

    .stButton > button {
        background-color: #4F46E5 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 500 !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease !important;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #4338CA !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1) !important;
    }

    .stDownloadButton > button {
        background-color: #10B981 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.2rem !important;
        width: 100%;
        margin-top: 1rem;
    }

    [data-testid="stImage"] {
        background-color: #ffffff;
        background-image:
            linear-gradient(45deg, #e5e5e5 25%, transparent 25%),
            linear-gradient(135deg, #e5e5e5 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, #e5e5e5 75%),
            linear-gradient(135deg, transparent 75%, #e5e5e5 75%);
        background-size: 20px 20px;
        background-position: 0 0, 10px 0, 10px -10px, 0px 10px;
        border-radius: 8px;
        overflow: hidden;
        border: 1px solid rgba(128, 128, 128, 0.2);
    }

    .image-card-title {
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        opacity: 0.8;
    }
    
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px;
        border: 2px dashed rgba(128, 128, 128, 0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Segformer API Logic
# ---------------------------------------------------------------------------
def hf_segformer_cutout(image_bytes: bytes, hf_token: str, grow_px: int) -> bytes:
    from huggingface_hub import InferenceClient
    
    if not hf_token:
        raise ValueError("Hugging Face API Token is missing.")
        
    # Initialize official client routed to the Segformer clothing model
    client = InferenceClient(model="mattmdjaga/segformer_b2_clothes", token=hf_token)
    
    # Run serverless image segmentation via the official SDK
    results = client.image_segmentation(image=image_bytes)
    
    source_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    target_labels = ["Upper-clothes", "Pants", "Skirt", "Dress", "Coat"]
    combined_mask = None
    
    for item in results:
        # Check standard label structures returned by InferenceClient
        label = item.get("label", "")
        if label in target_labels:
            mask_img = item.get("mask").convert("L")
            mask_img = mask_img.resize(source_img.size, Image.LANCZOS)
            
            if combined_mask is None:
                combined_mask = np.array(mask_img)
            else:
                combined_mask = np.maximum(combined_mask, np.array(mask_img))
                
    if combined_mask is None:
        raise ValueError("No garments detected by Segformer.")
        
    if grow_px > 0:
        kernel = np.ones((3, 3), np.uint8)
        combined_mask = cv2.erode(combined_mask, kernel, iterations=grow_px)
        
    mask_final = Image.fromarray(combined_mask)
    result = source_img.convert("RGBA")
    result.putalpha(mask_final)
    
    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()
# ---------------------------------------------------------------------------
# Local AI Cutout Logic
# ---------------------------------------------------------------------------
AI_MODELS = {
    "Clothes ONLY (Deletes Skin & Hair)": "u2net_cloth_seg",
    "U2Net Human (Best for People)": "u2net_human_seg",
    "IS-Net (Fine Edges & Details)": "isnet-general-use",
    "u2netp (Fast & Light)": "u2netp",
}

MAX_MASK_EDGE = 1024  

@st.cache_resource(show_spinner=False, max_entries=1)
def load_ai_session(model_name: str):
    import onnxruntime as ort
    from rembg.sessions import sessions_class
    opts = ort.SessionOptions()
    opts.enable_cpu_mem_arena = False
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session_cls = {cls.name(): cls for cls in sessions_class}[model_name]
    return session_cls(model_name, opts)

@st.cache_data(show_spinner=False, max_entries=2)
def ai_cutout(image_bytes: bytes, model_name: str, grow_px: int) -> bytes:
    from rembg import remove

    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    small = source.copy()
    small.thumbnail((MAX_MASK_EDGE, MAX_MASK_EDGE), Image.LANCZOS)
    
    if model_name == "u2net_cloth_seg":
        masks = remove(small, session=load_ai_session(model_name), only_mask=True, return_multiple=True)
        
        if isinstance(masks, list) and len(masks) >= 3:
            # Corrected Mapping based on visual artifacts: 
            # 0 = Face/Hair, 1 = Lower Body, 2 = Upper Body
            lower = np.array(masks[1].convert("L"))
            upper = np.array(masks[2].convert("L"))
            clothes_only = np.maximum(upper, lower)
            mask = Image.fromarray(clothes_only)
        elif isinstance(masks, list):
            mask = masks[0].convert("L")
        else:
            mask = masks.convert("L")
    else:
        raw_mask = remove(small, session=load_ai_session(model_name), only_mask=True)
        channels = raw_mask.split()
        mask = channels[-1] if len(channels) == 4 else raw_mask.convert("L")
        
    mask = mask.resize(source.size, Image.LANCZOS)

    if grow_px > 0:
        mask_arr = np.array(mask)
        kernel = np.ones((3, 3), np.uint8)
        mask_arr = cv2.erode(mask_arr, kernel, iterations=grow_px)
        mask = Image.fromarray(mask_arr)

    result = source.convert("RGBA")
    result.putalpha(mask)

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Chroma Key Logic
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
    hex_str = key_hex.lstrip('#')
    target_color = np.array([int(hex_str[i:i+2], 16) for i in (0, 2, 4)], dtype=np.float32)
    distances = np.linalg.norm(img_array.astype(np.float32) - target_color, axis=-1)
    
    mask = 1.0 - (distances - tola) / max(tolb - tola, 1)
    mask = np.clip(mask, 0.0, 1.0)
    
    alpha_f = 1.0 - mask
    final_rgb = img_array.copy()

    edge_mask = (alpha_f > 0.05) & (alpha_f < 0.95)
    if target_color[1] > target_color[0] and target_color[1] > target_color[2]:
        r, g, b = final_rgb[:,:,0].astype(np.float32), final_rgb[:,:,1].astype(np.float32), final_rgb[:,:,2].astype(np.float32)
        max_green = (r + b) / 2
        suppressed_g = np.where(edge_mask & (g > max_green), max_green, g)
        final_rgb[:,:,1] = suppressed_g.astype(np.uint8)

    if grow_px > 0:
        fg_binary = (alpha_f > 0.5).astype(np.uint8)
        core = cv2.erode(fg_binary, np.ones((3, 3), np.uint8), iterations=grow_px)
        if core.any():
            _, indices = distance_transform_edt(1 - core, return_indices=True)
            grown_rgb = final_rgb[indices[0], indices[1]]
            final_rgb = np.where(core[..., None].astype(bool), final_rgb, grown_rgb)

    alpha = np.uint8(np.clip(alpha_f, 0, 1) * 255)
    rgba = np.dstack([final_rgb, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


# ---------------------------------------------------------------------------
# UI - Main App Logic
# ---------------------------------------------------------------------------
st.markdown('<div class="app-header"><div class="app-title">Garment Extractor Pro</div><div class="app-subtitle">Create clean, transparent product assets instantly.</div></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload product photo to begin", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is None:
    st.info("👋 Upload a garment photo above to launch the extraction studio.")
    st.stop()

image_bytes = uploaded_file.getvalue()
original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
img_array = np.array(original_image)

with st.sidebar:
    st.markdown('<div class="sidebar-header">Engine Settings</div>', unsafe_allow_html=True)
    method = st.radio("Processing Engine", [
        "Segformer (Pro Garment AI)", 
        "Local AI (Basic & Full Subject)", 
        "Chroma Key (Studio Green)", 
        "Hybrid (AI + Chroma)"
    ], label_visibility="collapsed")
    
    st.markdown("---")
    
    if method == "Segformer (Pro Garment AI)":
        st.info("🚀 **State-of-the-Art Clothing AI.** Uses an advanced transformer API to map 18 garment classes while ignoring skin and backgrounds.")
        st.markdown('<div class="sidebar-header">API Configuration</div>', unsafe_allow_html=True)
        hf_token = st.text_input("Hugging Face API Token", type="password", help="Get a free API token from your huggingface.co account settings.")
        
    elif method == "Local AI (Basic & Full Subject)":
        st.info("💡 **Local Processing.** Extracts the general silhouette without an internet connection. Struggles with specific garments.")
        st.markdown('<div class="sidebar-header">AI Parameters</div>', unsafe_allow_html=True)
        model_label = st.selectbox("Model Tier", list(AI_MODELS), index=0, key="ai_model_sel")
        model_name = AI_MODELS[model_label]
            
    elif method == "Chroma Key (Studio Green)":
        st.warning("⚠️ **Best for solid backdrops only.** Relies on strict color contrast. Fails on complex backgrounds or shadows.")
        st.markdown('<div class="sidebar-header">Keying Parameters</div>', unsafe_allow_html=True)
        detected_hex = detect_background_hex(img_array)
        key_color_hex = st.color_picker("Key Color", detected_hex, key="chroma_color")
        tola = st.slider("Tolerance A (Shadows)", 1, 50, 10, key="chroma_tola")
        tolb = st.slider("Tolerance B (Highlights)", tola + 1, 120, 60, key="chroma_tolb")
                
    elif method == "Hybrid (AI + Chroma)":
        st.info("⚡ **Best for perfect green screens.** Combines AI shape detection with strict color math to punch out enclosed gaps.")
        st.markdown('<div class="sidebar-header">Hybrid Parameters</div>', unsafe_allow_html=True)
        model_label = st.selectbox("AI Model Tier", list(AI_MODELS), index=0, key="hyb_model_sel")
        model_name = AI_MODELS[model_label]
        detected_hex = detect_background_hex(img_array)
        key_color_hex = st.color_picker("Key Color", detected_hex, key="hyb_color")
        tola = st.slider("Tolerance A (Shadows)", 1, 50, 10, key="hyb_tola")
        tolb = st.slider("Tolerance B (Highlights)", tola + 1, 120, 60, key="hyb_tolb")
        
    with st.expander("Edge Cleanup", expanded=True):
        st.markdown("<p style='font-size: 0.9rem; font-weight: 500; margin-bottom: 0;'>Fringe Eraser (px)</p>", unsafe_allow_html=True)
        col_min, col_slide, col_plus = st.columns([1, 4, 1])
        with col_min:
            st.button("−", on_click=step_fringe, args=(-1,), width="stretch", key="btn_minus")
        with col_slide:
            grow_px = st.slider("Fringe", 0, 5, key="fringe_val", label_visibility="collapsed")
        with col_plus:
            st.button("+", on_click=step_fringe, args=(1,), width="stretch", key="btn_plus")


# 3. Process & Render Result
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="image-card-title">Source Image</div>', unsafe_allow_html=True)
    st.image(original_image, width="stretch")

with col2:
    st.markdown('<div class="image-card-title">Extraction Result</div>', unsafe_allow_html=True)
    
    try:
        if method == "Segformer (Pro Garment AI)":
            if not hf_token:
                st.warning("A Hugging Face API Token is required for this engine.")
                st.stop()
            with st.spinner("🧠 Querying Segformer Transformer..."):
                extracted_image = Image.open(io.BytesIO(hf_segformer_cutout(image_bytes, hf_token, grow_px)))

        elif method == "Local AI (Basic & Full Subject)":
            with st.spinner(f"🤖 Running {model_label.split(' ')[0]} model..."):
                extracted_image = Image.open(io.BytesIO(ai_cutout(image_bytes, model_name, grow_px)))
                
        elif method == "Chroma Key (Studio Green)":
            with st.spinner("🟩 Applying color math..."):
                extracted_image = chroma_cutout(img_array, key_color_hex, tola, tolb, grow_px)
                
        elif method == "Hybrid (AI + Chroma)":
            with st.spinner("⚔️ Fusing AI Shape & Color Math..."):
                ai_bytes = ai_cutout(image_bytes, model_name, 0)
                ai_alpha = np.array(Image.open(io.BytesIO(ai_bytes)).split()[-1])
                
                chroma_img = chroma_cutout(img_array, key_color_hex, tola, tolb, grow_px)
                chroma_alpha = np.array(chroma_img.split()[-1])
                
                combined_alpha = np.minimum(ai_alpha, chroma_alpha)
                
                extracted_image = chroma_img.copy()
                extracted_image.putalpha(Image.fromarray(combined_alpha))

        st.image(extracted_image, width="stretch")
        
        buf = io.BytesIO()
        extracted_image.save(buf, format="PNG")
        st.download_button(
            label="↓ Export Transparent Asset (PNG)",
            data=buf.getvalue(),
            file_name="product_asset.png",
            mime="image/png",
            width="stretch"
        )
        
    except ModuleNotFoundError as exc:
        st.error(f"**Missing Engine Dependency:** `{exc.name}`")
        st.info("Check your `requirements.txt` file in Streamlit Cloud.")
    except Exception as exc: 
        st.error(f"**Processing Error:** {exc}")
