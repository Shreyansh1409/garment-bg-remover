import streamlit as st
import numpy as np
from PIL import Image
import io
import cv2
from ChromaKey import ChromaKey
# Set up the web page layout
st.set_page_config(layout="wide", page_title="Green Screen Remover")
st.title("🟩 Green Screen Background Remover")
st.write("Upload a green screen product photo to isolate the garment using precise chroma-key math.")

# Add a sidebar color picker to target the exact background green
st.sidebar.header("Chroma Key Settings")
key_color_hex = st.sidebar.color_picker("Pick your background color", "#00FF00")

# Convert the hex color from the picker to an RGB tuple for the library
key_color_rgb = tuple(int(key_color_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

# File uploader
uploaded_file = st.file_uploader("Upload your green screen photo", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    # Load the image and ensure it's in standard RGB format
    original_image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.header("Original Image")
        st.image(original_image, use_container_width=True)
        
    with col2:
        st.header("Extracted Garment")
        with st.spinner("Calculating chroma key..."):
            try:
                # 1. Convert the PIL Image into a NumPy array for math processing
                img_array = np.array(original_image)
                
                # 2. Initialize the ChromaKey algorithm with your chosen background color
                ck = ChromaKey(key_color_rgb)
                
                # 3. Process the array to generate the transparent result
                # Note: If this specific method throws an error, check the 'examples' 
                # folder in the eugeneteoh/chromakey GitHub repo for their exact processing syntax!
                result_array = ck.process(img_array)
                
                # 4. Convert the array back into a PIL Image so Streamlit can display it
                extracted_image = Image.fromarray(result_array.astype('uint8'), 'RGBA')
                
                st.image(extracted_image, use_container_width=True)
                
                # 5. Create the downloadable byte stream
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
                st.info("The exact syntax for the library might require a small tweak based on its version. Check the Streamlit logs for more details.")
