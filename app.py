import streamlit as st
import numpy as np
from PIL import Image
import io
import cv2
from chromakey import ChromaKey 

st.set_page_config(layout="wide", page_title="Green Screen Remover")
st.title("🟩 Green Screen Background Remover")
st.write("Upload a green screen product photo to isolate the garment using precise chroma-key math.")

st.sidebar.header("Chroma Key Settings")
key_color_hex = st.sidebar.color_picker("Pick your background color", "#00FF00")
key_color_rgb = tuple(int(key_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

uploaded_file = st.file_uploader("Upload your green screen photo", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    original_image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.header("Original Image")
        st.image(original_image, use_container_width=True)
        
    with col2:
        st.header("Extracted Garment")
        with st.spinner("Calculating chroma key..."):
            try:
                img_array = np.array(original_image)
                
                # Initialize ChromaKey from the installed GitHub repository
                ck = ChromaKey(key_color_rgb)
                result_array = ck.process(img_array)
                
                extracted_image = Image.fromarray(result_array.astype('uint8'), 'RGBA')
                st.image(extracted_image, use_container_width=True)
                
                buf = io.BytesIO()
                extracted_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="Download Transparent PNG",
                    data=byte_im,
                    file_name="chromakey_cutout.png",
                    mime="image/png",
                    use_container_width=True
                )
            except Exception as e:
                st.error(f"Error processing image: {e}")
