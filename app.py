import io

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from scipy.ndimage import distance_transform_edt

# ---------------------------------------------------------------------------
# App Configuration & Modern App CSS
# ---------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="Garment Extractor Pro", page_icon="✂️", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* Global App Font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        background-color: #F9FAFB;
        color: #111827;
    }

    /* Clean up top padding and hide Streamlit branding */
    .block-container { padding-top: 2rem !important; max-width: 1400px; }
    header { visibility: hidden; }
    footer { visibility: hidden; }

    /* App Header Styling */
    .app-header {
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #E5E7EB;
    }
    .app-title {
        font-weight: 700;
        font-size: 2.25rem;
        color: #111827;
        margin-bottom: 0.25rem;
    }
    .app-subtitle {
        font-size: 1rem;
        color: #6B7280;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
    .sidebar-header {
        font-weight: 600;
        font-size: 1.1rem;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
        color: #374151;
    }

    /* Primary Buttons (Process / Clear) */
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

    /* Download Button (Success state) */
    .stDownloadButton > button {
        background-color: #10B981 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        font-weight: 600 !important;
        padding: 0.75rem 1.2rem !important;
        transition: all 0.2s ease !important;
        width: 100%;
        margin-top: 1rem;
    }
    .stDownloadButton > button:hover {
        background-color: #059669 !important;
        box-shadow: 0 4px 6px -1px rgba(16, 185, 129, 0.2) !important;
    }

    /* Image Card Containers */
    .image-card {
        background: #FFFFFF;
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        border: 1px solid #E5E7EB;
        margin-bottom: 1rem;
    }
    .image-card-title {
        font-weight: 600;
        font-size: 0.9rem;
        color: #4B5563;
        margin-bottom: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Uploader styling */
    [data-testid="stFileUploaderDropzone"] {
        border-radius: 12px;
        border: 2px dashed #D1D5DB;
        background-color: #FFFFFF;
        transition: all 0.2s;
    }
    [data-testid="stFileUploaderDropzone"]:hover {
        border-color: #4F46E5;
        background-color: #F5F3FF;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Core Logic: AI Cutout
# ---------------------------------------------------------------------------
AI_MODELS = {
    "u2netp (Fast/Light)": "u2netp",
    "u2net (High Quality)": "u2net",
}

MAX_MASK_EDGE = 1024  

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
def ai_cutout(image_bytes: bytes, model_name: str, grow_px: int) -> bytes:
    from rembg import remove

    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    small = source.copy()
    small.thumbnail((MAX_MASK_EDGE, MAX_MASK_EDGE), Image.LANCZOS)
    
    raw_mask = remove(small, session=load_ai_session(model_name), only_mask=True)
    channels = raw_mask.split()
    mask = channels[-1] if len(channels) == 4 else raw_mask.convert("L")
    mask = mask.resize(source.size, Image.LANCZOS)

    # Fringe Removal (Erosion)
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
# Core Logic: Chroma Key
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

    # Spill Suppression
    edge_mask = (alpha_f > 0.05) & (alpha_f < 0.95)
    if target_color[1] > target_color[0] and target_color[1] > target_color[2]:
        r, g, b = final_rgb[:,:,0].astype(np.float32), final_rgb[:,:,1].astype(np.float32), final_rgb[:,:,2].astype(np.float32)
        max_green = (r + b) / 2
        suppressed_g = np.where(edge_mask & (g > max_green), max_green, g)
        final_rgb[:,:,1] = suppressed_g.astype(np.uint8)

    # Fringe removal
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
# UI - App Shell & Sidebar
# ---------------------------------------------------------------------------
st.markdown('<div class="app-header"><div class="app-title">Garment Extractor Pro</div><div class="app-subtitle">Create clean, transparent product assets instantly.</div></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<div class="sidebar-header">Engine Settings</div>', unsafe_allow_html=True)
    method = st.radio("Processing Engine", ["AI Engine (Auto)", "Chroma Key (Studio Green)"], label_visibility="collapsed")
    
    st.markdown("---")
    
    if method == "AI Engine (Auto)":
        st.markdown('<div class="sidebar-header">AI Parameters</div>', unsafe_allow_html=True)
        model_label = st.selectbox("Model Tier", list(AI_MODELS), index=0)
        model_name = AI_MODELS[model_label]
        
        with st.expander("Advanced Output Settings"):
            grow_px = st.slider("Fringe Eraser (px)", 0, 5, 0, help="Shrinks the cutout mask to remove edge halos.")
            
    else:
        st.markdown('<div class="sidebar-header">Keying Parameters</div>', unsafe_allow_html=True)
        detected_hex = "#3A6047" # Default placeholder, gets updated below
        key_color_hex = st.color_picker("Key Color", detected_hex)
        tola = st.slider("Tolerance A (Shadows)", 1, 50, 10, help="Pixels darker/closer to this color are deleted.")
        tolb = st.slider("Tolerance B (Highlights)", tola + 1, 120, 60, help="Pixels beyond this distance are kept 100%.")
        
        with st.expander("Edge Cleanup"):
            grow_px = st.slider("Fringe Eraser (px)", 0, 5, 2)


# ---------------------------------------------------------------------------
# UI - Main Workspace
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader("Upload product photo to begin", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file is None:
    st.info("👋 Upload a garment photo above to launch the extraction studio.")
    st.stop()

# Load image
image_bytes = uploaded_file.getvalue()
original_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
img_array = np.array(original_image)

# Update sidebar chroma color dynamically if chroma is selected
if method == "Chroma Key (Studio Green)":
    detected_hex = detect_background_hex(img_array)
    # Note: Streamlit color_picker doesn't auto-update from script flow easily without session state, 
    # but the user will see their image and can pick the correct color.

# Display Layout
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div class="image-card">
            <div class="image-card-title">Source Image</div>
        </div>
        """, unsafe_allow_html=True
    )
    st.image(original_image, use_column_width=True)

with col2:
    st.markdown(
        """
        <div class="image-card">
            <div class="image-card-title">Extraction Result</div>
        </div>
        """, unsafe_allow_html=True
    )
    
    try:
        if method == "AI Engine (Auto)":
            with st.spinner("🤖 Analyzing garment topology..."):
                extracted_image = Image.open(io.BytesIO(ai_cutout(image_bytes, model_name, grow_px)))
        else:
            with st.spinner("🟩 Applying color math..."):
                extracted_image = chroma_cutout(img_array, key_color_hex, tola, tolb, grow_px)

        st.image(extracted_image, use_column_width=True)
        
        buf = io.BytesIO()
        extracted_image.save(buf, format="PNG")
        st.download_button(
            label="↓ Export Transparent Asset (PNG)",
            data=buf.getvalue(),
            file_name="product_asset.png",
            mime="image/png"
        )
        
    except ModuleNotFoundError as exc:
        st.error(f"**Missing Engine Dependency:** `{exc.name}`")
        st.info("Check your `requirements.txt` file in Streamlit Cloud.")
    except Exception as exc: 
        st.error(f"**Processing Error:** {exc}")
