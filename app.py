import streamlit as st
from rembg import remove, new_session
from PIL import Image
import io

st.set_page_config(layout="wide", page_title="Apparel Background Remover")
st.title("👕 Garment Background Remover")
st.write("Upload a product photo to isolate the clothing using garment-specific AI.")

@st.cache_resource
def load_model():
    # Loads the specific AI model trained to recognize clothing
    return new_session("u2net_cloth_seg")

session = load_model()

uploaded_file = st.file_uploader("Upload your garment photo", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    original_image = Image.open(uploaded_file)
    
    with col1:
        st.header("Original Image")
        st.image(original_image, use_column_width=True)
        
    with col2:
        st.header("Extracted Garment")
        with st.spinner("Isolating clothing..."):
            extracted_image = remove(original_image, session=session)
            st.image(extracted_image, use_column_width=True)
            
            buf = io.BytesIO()
            extracted_image.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download Transparent PNG",
                data=byte_im,
                file_name="garment_cutout.png",
                mime="image/png",
                use_container_width=True
            )
