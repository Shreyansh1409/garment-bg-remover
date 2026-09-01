import base64
import io
import os
import cv2
import numpy as np
import requests
import streamlit as st
from PIL import Image
from scipy.ndimage import distance_transform_edt

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

if "reconstructed_image" not in st.session_state:
    st.session_state.reconstructed_image = None

def step_fringe(delta: int):
    st.session_state.fringe_val = max(0, min(5, st.session_state.fringe_val + delta))

# ---------------------------------------------------------------------------
# Modern App CSS Styling
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
        margin-top: 0.75rem;
    }
    .stDownloadButton > button:hover {
        background-color: #059669 !important;
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
# Segformer Local PyTorch Engine
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False, max_entries=1)
def load_segformer():
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor
    processor = SegformerImageProcessor.from_pretrained("mattmdjaga/segformer_b2_clothes")
    model = AutoModelForSemanticSegmentation.from_pretrained("mattmdjaga/segformer_b2_clothes")
    return processor, model

@st.cache_data(show_spinner=False, max_entries=2)
def local_segformer_cutout(image_bytes: bytes, grow_px: int):
    import torch
    import torch.nn as nn
    
    processor, model = load_segformer()
    source_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    
    inputs = processor(images=source_img, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        
    logits = outputs.logits.cpu()
    upsampled_logits = nn.functional.interpolate(
        logits,
        size=source_img.size[::-1],
        mode="bilinear",
        align_corners=False,
    )
    
    pred_seg = upsampled_logits.argmax(dim=1)[0].numpy()
    
    # Target clothing labels: Upper-clothes (4), Skirt (5), Pants (6), Dress (7)
    clothing_labels = [4, 5, 6, 7]
    garment_mask_arr = np.isin(pred_seg, clothing_labels).astype(np.uint8) * 255
    
    # Occlusion labels: Hair (2), Face (11), Left Arm (14), Right Arm (15)
    occlusion_labels = [2, 11, 14, 15]
    occlusion_mask_arr = np.isin(pred_seg, occlusion_labels).astype(np.uint8) * 255
    
    if grow_px > 0:
        kernel = np.ones((3, 3), np.uint8)
        garment_mask_arr = cv2.erode(garment_mask_arr, kernel, iterations=grow_px)
        occlusion_mask_arr = cv2.dilate(occlusion_mask_arr, kernel, iterations=grow_px)
        
    garment_mask = Image.fromarray(garment_mask_arr, mode="L")
    inpaint_mask = Image.fromarray(occlusion_mask_arr, mode="L")
    
    cutout = source_img.convert("RGBA")
    cutout.putalpha(garment_mask)
    
    garment_buf = io.BytesIO()
    cutout.save(garment_buf, format="PNG")
    
    mask_buf = io.BytesIO()
    inpaint_mask.save(mask_buf, format="PNG")
    
    return garment_buf.getvalue(), mask_buf.getvalue()

# ---------------------------------------------------------------------------
# Cloud Inpainting API Engine
# ---------------------------------------------------------------------------
def api_inpaint_fabric(source_bytes: bytes, mask_bytes: bytes, token: str) -> bytes:
    endpoints = [
        "https://router.huggingface.co/hf-inference/models/runwayml/stable-diffusion-inpainting",
        "https://api-inference.huggingface.co/models/runwayml/stable-diffusion-inpainting"
    ]
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": "seamless continuous garment fabric, smooth clean texture, studio clothing flat lay",
        "image": base64.b64encode(source_bytes).decode("utf-8"),
        "mask_image": base64.b64encode(mask_bytes).decode("utf-8"),
        "parameters": {
            "negative_prompt": "skin, hair, arms, hands, human body parts, noisy artifacts, blurry",
            "guidance_scale": 7.5,
            "strength": 0.99
        }
    }
    
    last_error = None
    for url in endpoints:
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=45)
            if response.status_code == 200:
                return response.content
            last_error = f"Status {response.status_code}: {response.text}"
        except Exception as e:
            last_error = str(e)
            
    raise RuntimeError(f"Cloud Inpainting service failed. Details: {last_error}")

# ---------------------------------------------------------------------------
# Local AI (rembg) Engine
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
# Chroma Key Engine
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
# App Interface & Layout
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="app-header">'
    '<div class="app-title">Garment Extractor Pro</div>'
    '<div class="app-subtitle">Create clean, transparent product assets instantly.</div>'
    '</div>', 
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload product photo to begin", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is None:
    st.info("Upload a garment photo above to launch the extraction studio.")
    st.stop()

image_bytes = uploaded_file.getvalue()
original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
img_array = np.array(original_image)

auto_token = ""
try:
    if hasattr(st, "secrets") and "HUGGINGFACE_TOKEN" in st.secrets:
        auto_token = str(st.secrets["HUGGINGFACE_TOKEN"])
except Exception:
    pass
if not auto_token:
    auto_token = os.getenv("HUGGINGFACE_TOKEN", "")

with st.sidebar:
    st.markdown('<div class="sidebar-header">Engine Settings</div>', unsafe_allow_html=True)
    method = st.radio(
        "Processing Engine",
        [
            "Segformer (Pro Garment AI)", 
            "Local AI (Basic & Full Subject)", 
            "Chroma Key (Studio Green)", 
            "Hybrid (AI + Chroma)"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    if method == "Segformer (Pro Garment AI)":
        st.info("Accurately maps clothing classes while deleting hair, arms, and skin. Supports one-click cloud inpainting to fill occlusion gaps.")
        st.markdown('<div class="sidebar-header">API Configuration</div>', unsafe_allow_html=True)
        hf_token = st.text_input("Hugging Face API Token", value=auto_token, type="password")
        
    elif method == "Local AI (Basic & Full Subject)":
        st.info("Extracts the full subject silhouette locally using standard segmentation models.")
        st.markdown('<div class="sidebar-header">AI Parameters</div>', unsafe_allow_html=True)
        model_label = st.selectbox("Model Tier", list(AI_MODELS), index=0, key="ai_model_sel")
        model_name = AI_MODELS[model_label]
            
    elif method == "Chroma Key (Studio Green)":
        st.warning("Performs mathematical background subtraction based on key color distance.")
        st.markdown('<div class="sidebar-header">Keying Parameters</div>', unsafe_allow_html=True)
        detected_hex = detect_background_hex(img_array)
        key_color_hex = st.color_picker("Key Color", detected_hex, key="chroma_color")
        tola = st.slider("Tolerance A (Shadows)", 1, 50, 10, key="chroma_tola")
        tolb = st.slider("Tolerance B (Highlights)", tola + 1, 120, 60, key="chroma_tolb")
                
    elif method == "Hybrid (AI + Chroma)":
        st.info("Combines AI silhouette detection with chroma key color math to punch out inner gaps.")
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
            st.button("−", on_click=step_fringe, args=(-1,), key="btn_minus")
        with col_slide:
            grow_px = st.slider("Fringe", 0, 5, key="fringe_val", label_visibility="collapsed")
        with col_plus:
            st.button("+", on_click=step_fringe, args=(1,), key="btn_plus")

# ---------------------------------------------------------------------------
# Processing & Rendering Columns
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="image-card-title">Source Image</div>', unsafe_allow_html=True)
    st.image(original_image, use_container_width=True)

with col2:
    st.markdown('<div class="image-card-title">Extraction Result</div>', unsafe_allow_html=True)
    
    try:
        if method == "Segformer (Pro Garment AI)":
            with st.spinner("Extracting garments and occlusion zones..."):
                garment_bytes, mask_bytes = local_segformer_cutout(image_bytes, grow_px)
                extracted_image = Image.open(io.BytesIO(garment_bytes))
                inpaint_mask = Image.open(io.BytesIO(mask_bytes))
                
            st.image(extracted_image, use_container_width=True)
            
            st.download_button(
                label="↓ Export Isolated Garment (PNG)",
                data=garment_bytes,
                file_name="extracted_garment.png",
                mime="image/png",
                use_container_width=True
            )
            
            st.markdown("---")
            st.markdown('<div class="image-card-title">Generative Inpainting Reconstruction</div>', unsafe_allow_html=True)
            st.caption("Reconstruct fabric covered by hair or arms by sending the occlusion mask to the cloud inpainting engine.")
            
            if st.button("✨ Reconstruct Missing Fabric", use_container_width=True):
                if not hf_token:
                    st.warning("Enter your Hugging Face API Token in the sidebar to enable generative inpainting.")
                else:
                    with st.spinner("Reconstructing missing fabric via Cloud API..."):
                        try:
                            reconstructed_bytes = api_inpaint_fabric(image_bytes, mask_bytes, hf_token)
                            st.session_state.reconstructed_image = reconstructed_bytes
                        except Exception as e:
                            st.error(f"Inpainting Error: {e}")
                            
            if st.session_state.reconstructed_image is not None:
                recon_img = Image.open(io.BytesIO(st.session_state.reconstructed_image))
                st.image(recon_img, use_container_width=True)
                st.download_button(
                    label="↓ Export Final Reconstructed Garment (PNG)",
                    data=st.session_state.reconstructed_image,
                    file_name="reconstructed_garment.png",
                    mime="image/png",
                    use_container_width=True
                )
                            
        else:
            with st.spinner("Processing cutout..."):
                if method == "Local AI (Basic & Full Subject)":
                    out_bytes = ai_cutout(image_bytes, model_name, grow_px)
                    extracted_image = Image.open(io.BytesIO(out_bytes))
                elif method == "Chroma Key (Studio Green)":
                    extracted_image = chroma_cutout(img_array, key_color_hex, tola, tolb, grow_px)
                elif method == "Hybrid (AI + Chroma)":
                    ai_bytes = ai_cutout(image_bytes, model_name, 0)
                    ai_alpha = np.array(Image.open(io.BytesIO(ai_bytes)).split()[-1])
                    chroma_img = chroma_cutout(img_array, key_color_hex, tola, tolb, grow_px)
                    chroma_alpha = np.array(chroma_img.split()[-1])
                    combined_alpha = np.minimum(ai_alpha, chroma_alpha)
                    extracted_image = chroma_img.copy()
                    extracted_image.putalpha(Image.fromarray(combined_alpha))
                    
            st.image(extracted_image, use_container_width=True)
            
            buf = io.BytesIO()
            extracted_image.save(buf, format="PNG")
            st.download_button(
                label="↓ Export Transparent Asset (PNG)",
                data=buf.getvalue(),
                file_name="product_asset.png",
                mime="image/png",
                use_container_width=True
            )
            
    except ModuleNotFoundError as exc:
        st.error(f"Missing Engine Dependency: `{exc.name}`")
        st.info("Ensure the dependency is listed in your `requirements.txt` file.")
    except Exception as exc: 
        st.error(f"Processing Error: {exc}")
