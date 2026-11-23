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
    initial_sidebar_state="expanded"
)

if "fringe_val" not in st.session_state:
    st.session_state.fringe_val = 0

def step_fringe(delta: int):
    st.session_state.fringe_val = max(0, min(5, st.session_state.fringe_val + delta))

# ---------------------------------------------------------------------------
# Modern App CSS Styling (Clean Light Theme)
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] { 
        font-family: 'Inter', sans-serif; 
    }
    
    /* Base App Theme - Clean Light */
    .stApp { 
        background-color: #f8fafc; 
    }
    
    /* Layout Spacing */
    .block-container { 
        padding-top: 2rem !important; 
        max-width: 1400px; 
    }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    
    /* Header Typography */
    .app-header { 
        margin-bottom: 2.5rem; 
        padding-bottom: 1.5rem; 
        border-bottom: 1px solid rgba(0, 0, 0, 0.05); 
    }
    .app-title { 
        font-weight: 800; 
        font-size: 2.8rem; 
        letter-spacing: -0.03em; 
        margin-bottom: 0.25rem;
        color: #0f172a;
    }
    .app-subtitle { 
        font-size: 1.05rem; 
        color: #64748b; 
        font-weight: 400;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid rgba(0, 0, 0, 0.05);
    }
    [data-testid="stSidebar"] * {
        color: #334155 !important;
    }
    .sidebar-header { 
        font-weight: 700; 
        font-size: 0.85rem; 
        color: #94a3b8 !important;
        text-transform: uppercase; 
        letter-spacing: 0.05em;
        margin: 1.5rem 0 0.75rem 0; 
    }
    
    /* Primary Buttons (Sleek Dark) */
    .stButton > button { 
        background-color: #0f172a !important;
        color: white !important; 
        border-radius: 8px !important; 
        border: none !important; 
        font-weight: 500 !important; 
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05) !important;
    }
    .stButton > button p, .stDownloadButton > button p { color: white !important; }
    .stButton > button:hover {
        background-color: #334155 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Download Buttons (Vibrant Action) */
    .stDownloadButton > button { 
        background-color: #2563eb !important;
        color: white !important; 
        border-radius: 8px !important; 
        border: none !important; 
        font-weight: 600 !important; 
        padding: 0.8rem 1.2rem !important;
        margin-top: 1rem;
        transition: all 0.2s ease !important;
        box-shadow: 0 4px 6px rgba(37, 99, 235, 0.2) !important;
    }
    .stDownloadButton > button:hover { 
        background-color: #1d4ed8 !important;
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(37, 99, 235, 0.3) !important;
    }
    
    /* Image Cards & Text */
    .image-card-title { 
        font-weight: 600; 
        font-size: 0.85rem; 
        color: #64748b;
        margin-bottom: 0.75rem; 
        text-transform: uppercase; 
        letter-spacing: 0.05em;
    }
    p, span, label, div[data-baseweb="radio"] {
        color: #334155;
    }
    
    /* Upload Dropzone */
    [data-testid="stFileUploaderDropzone"] { 
        border-radius: 12px; 
        border: 2px dashed #cbd5e1; 
        background-color: #ffffff;
        transition: all 0.3s ease;
    }
    [data-testid="stFileUploaderDropzone"]:hover { 
        border-color: #2563eb; 
        background-color: #f8fafc;
    }
    
    /* White/Light Gray Transparency Grid */
    [data-testid="stImage"] {
        background-color: #ffffff !important;
        background-image:
            linear-gradient(45deg, #f1f5f9 25%, transparent 25%),
            linear-gradient(135deg, #f1f5f9 25%, transparent 25%),
            linear-gradient(45deg, transparent 75%, #f1f5f9 75%),
            linear-gradient(135deg, transparent 75%, #f1f5f9 75%) !important;
        background-size: 16px 16px !important;
        background-position: 0 0, 8px 0, 8px -8px, 0px 8px !important;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
        overflow: hidden;
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
    
    clothing_labels = [4, 5, 6, 7]
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
    
    session_dict = {cls.name(): cls for cls in sessions_class}
    return session_dict["u2net_cloth_seg"]("u2net_cloth_seg", opts)

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
# 3D Generation (offloaded to hosted GPU Spaces via gradio_client)
# ---------------------------------------------------------------------------
# This Space runs on CPU, so image-to-3D runs on public ZeroGPU Spaces.
# Anonymous callers share a small GPU quota that is often exhausted; set an
# HF_TOKEN secret on this Space so calls count against your own quota.
ENGINES_3D = {
    "TRELLIS (textured, ~30s)": "trellis-community/TRELLIS",
    "Hunyuan3D-2 (shape only, no texture)": "tencent/Hunyuan3D-2",
}

def prepare_for_3d(cutout: Image.Image, margin: float = 0.1) -> bytes:
    """Crop the garment to its alpha bounding box and centre it on a square transparent canvas."""
    rgba = cutout.convert("RGBA")
    bbox = rgba.getchannel("A").point(lambda a: 255 if a > 10 else 0).getbbox()
    if bbox is None:
        raise ValueError("The extraction is empty - there is no garment to turn into 3D.")
    rgba = rgba.crop(bbox)
    side = int(max(rgba.size) * (1 + 2 * margin))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(rgba, ((side - rgba.width) // 2, (side - rgba.height) // 2), rgba)
    canvas.thumbnail((1024, 1024), Image.LANCZOS)
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()

def _file_path(value):
    """Gradio returns either a plain path or an update dict like {'value': path}."""
    if isinstance(value, dict):
        return value.get("value") or value.get("path")
    return value

@st.cache_data(show_spinner=False, max_entries=4)
def generate_3d(png_bytes: bytes, engine: str) -> bytes:
    import os
    import tempfile
    from gradio_client import Client, handle_file

    token = os.environ.get("HF_TOKEN") or None
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(png_bytes)
        input_path = tmp.name
    try:
        client = Client(ENGINES_3D[engine], token=token, verbose=False)
        if engine.startswith("TRELLIS"):
            # The UI normally fires these on page load / upload; API callers must do it explicitly.
            client.predict(api_name="/start_session")  # creates the per-session output folder
            prepped = client.predict(handle_file(input_path), api_name="/preprocess_image")  # keeps our alpha
            _, _, glb_path = client.predict(
                image=handle_file(_file_path(prepped)),
                multiimages=[],
                seed=0,
                ss_guidance_strength=7.5,
                ss_sampling_steps=12,
                slat_guidance_strength=3.0,
                slat_sampling_steps=12,
                multiimage_algo="stochastic",
                mesh_simplify=0.95,
                texture_size=1024,
                api_name="/generate_and_extract_glb",
            )
        else:
            # The public Space's texture stage is broken, so only the shape endpoint is usable.
            glb_path, *_ = client.predict(
                caption=None,
                image=handle_file(input_path),
                mv_image_front=None, mv_image_back=None, mv_image_left=None, mv_image_right=None,
                steps=30,
                guidance_scale=5.0,
                seed=1234,
                octree_resolution=256,
                check_box_rembg=False,  # already cut out; re-running rembg can eat garment edges
                num_chunks=8000,
                randomize_seed=False,
                api_name="/shape_generation",
            )
    finally:
        os.unlink(input_path)

    with open(_file_path(glb_path), "rb") as fh:
        return fh.read()

def explain_3d_error(exc: Exception) -> str:
    import os
    msg = str(exc)
    if "quota" in msg.lower() or "runs limit" in msg.lower() or "upstream gradio app has raised" in msg.lower():
        hint = ("Add an `HF_TOKEN` secret (Space → Settings → Variables and secrets) so calls use your own GPU quota."
                if not os.environ.get("HF_TOKEN") else
                "Your Hugging Face GPU quota may be used up for now - wait and retry, or switch engine.")
        return f"The hosted 3D engine refused the job. {hint}\n\nDetails: {msg}"
    return f"3D generation failed: {msg}"

def render_glb_viewer(glb_bytes: bytes, height: int = 520):
    import base64
    import streamlit.components.v1 as components
    b64 = base64.b64encode(glb_bytes).decode()
    components.html(
        f"""
        <script type="module" src="https://ajax.googleapis.com/ajax/libs/model-viewer/3.5.0/model-viewer.min.js"></script>
        <model-viewer src="data:model/gltf-binary;base64,{b64}" camera-controls auto-rotate
            shadow-intensity="1" exposure="1" environment-image="neutral"
            style="width:100%;height:{height - 20}px;background:#f1f5f9;border-radius:12px;">
        </model-viewer>
        """,
        height=height,
    )

# ---------------------------------------------------------------------------
# App Interface
# ---------------------------------------------------------------------------
st.markdown('<div class="app-header"><div class="app-title">TexAI Extractor</div><div class="app-subtitle">High-Fidelity Garment Isolation Engine. Designed to instantly generate clean catalog assets.</div></div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Upload product photo to begin", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
if uploaded_file is None:
    st.info("Upload a garment photo above to launch the extraction studio.")
    st.stop()

image_bytes = uploaded_file.getvalue()
original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
img_array = np.array(original_image)

with st.sidebar:
    st.markdown('<div class="sidebar-header">Extraction Engine</div>', unsafe_allow_html=True)
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
        st.info("⚡ **Combines AI with precise Chroma math.** Punches out enclosed background gaps and refines edges.")
        repair_occlusion = st.checkbox("Rebuild fabric behind arms", value=False)
        detected_hex = detect_background_hex(img_array)
        key_color_hex = st.color_picker("Key Color", detected_hex)
        tola = st.slider("Shadow Tolerance", 1, 50, 10)
        tolb = st.slider("Highlight Tolerance", tola + 1, 120, 60)
        
    with st.expander("Edge Cleanup", expanded=True):
        col_min, col_slide, col_plus = st.columns([1, 4, 1])
        with col_min: st.button("−", on_click=step_fringe, args=(-1,), width="stretch")
        with col_slide: grow_px = st.slider("Fringe", 0, 5, key="fringe_val", label_visibility="collapsed")
        with col_plus: st.button("+", on_click=step_fringe, args=(1,), width="stretch")

# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown('<div class="image-card-title">Source Image</div>', unsafe_allow_html=True)
    st.image(original_image, width="stretch")

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
                ai_bytes = u2net_cloth_cutout(image_bytes, repair_occlusion, grow_px)
                ai_alpha = np.array(Image.open(io.BytesIO(ai_bytes)).split()[-1])
                
                chroma_img = chroma_cutout(img_array, key_color_hex, tola, tolb, grow_px)
                chroma_alpha = np.array(chroma_img.split()[-1])
                
                combined_alpha = np.minimum(ai_alpha, chroma_alpha)
                
                if grow_px > 0:
                    combined_alpha = cv2.erode(combined_alpha, np.ones((3, 3), np.uint8), iterations=grow_px)
                    
                extracted_image = chroma_img.copy()
                extracted_image.putalpha(Image.fromarray(combined_alpha))
                
        st.image(extracted_image, width="stretch")
        
        buf = io.BytesIO()
        extracted_image.save(buf, format="PNG")
        st.download_button("↓ Export Transparent Garment", data=buf.getvalue(), file_name="garment_asset.png", mime="image/png", width="stretch")
            
    except Exception as exc:
        st.error(f"Processing Error: {exc}")
        extracted_image = None

# ---------------------------------------------------------------------------
# 3D Studio
# ---------------------------------------------------------------------------
if extracted_image is not None:
    st.markdown("---")
    st.markdown('<div class="image-card-title">3D Model</div>', unsafe_allow_html=True)
    st.caption("Turns the extracted garment into a textured, rotatable 3D model (.glb). "
               "The back and sides are AI-generated guesses - a single photo cannot show them.")

    col_engine, col_go = st.columns([3, 1])
    with col_engine:
        engine_3d = st.selectbox("3D Engine", list(ENGINES_3D), label_visibility="collapsed")
    with col_go:
        go_3d = st.button("✨ Generate 3D", width="stretch", type="primary")

    try:
        input_3d = prepare_for_3d(extracted_image)
        job_key = (hash(input_3d), engine_3d)
        if go_3d:
            with st.spinner("Building 3D model on a hosted GPU (20s – 2 min)..."):
                st.session_state.glb = generate_3d(input_3d, engine_3d)
                st.session_state.glb_key = job_key

        if st.session_state.get("glb") and st.session_state.get("glb_key") == job_key:
            render_glb_viewer(st.session_state.glb)
            st.download_button("↓ Export 3D Model (.glb)", data=st.session_state.glb,
                               file_name="garment_model.glb", mime="model/gltf-binary", width="stretch")
        elif st.session_state.get("glb"):
            st.info("Extraction settings changed - press Generate 3D to rebuild the model.")
    except Exception as exc:
        st.error(explain_3d_error(exc))
