import streamlit as st
import numpy as np
from PIL import Image
import io
import cv2
import chromakey

st.set_page_config(layout="wide", page_title="Green Screen Remover")
st.title("🟩 Green Screen Background Remover")
st.write("Upload a green screen product photo to isolate the garment using chroma keying.")

st.sidebar.header("Chroma Key Settings")
key_color_hex = st.sidebar.color_picker("Pick your background color", "#00FF00")

key_color_rgb = [int(key_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)]

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
                
                if hasattr(chromakey, 'chroma_key'):
                    result_array = chromakey.chroma_key(img_array, key_color_rgb)
                elif hasattr(chromakey, 'green_screen'):
                    result_array = chromakey.green_screen(img_array)
                else:
                    hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                    target_pixel = np.uint8([[key_color_rgb]])
                    target_hsv = cv2.cvtColor(target_pixel, cv2.COLOR_RGB2HSV)[0][0]
                    
                    lower_bound = np.array([max(0, target_hsv[0] - 25), 40, 40])
                    upper_bound = np.array([min(179, target_hsv[0] + 25), 255, 255])
                    
                    mask = cv2.inRange(hsv, lower_bound, upper_bound)
                    
                    rgba = cv2.cvtColor(img_array, cv2.COLOR_RGB2RGBA)
                    rgba[mask > 0] = [0, 0, 0, 0]
                    result_array = rgba

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
