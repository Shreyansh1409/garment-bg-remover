@st.cache_data(show_spinner=False, max_entries=4)
def ai_cutout(image_bytes: bytes, model_name: str, grow_px: int) -> bytes:
    from rembg import remove

    source = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    small = source.copy()
    small.thumbnail((MAX_MASK_EDGE, MAX_MASK_EDGE), Image.LANCZOS)
    
    raw_mask = remove(small, session=load_ai_session(model_name), only_mask=True)
    
    if model_name == "u2net_cloth_seg":
        # Force to RGB to isolate specific garment channels
        rgb_mask = raw_mask.convert("RGB")
        r, g, b = rgb_mask.split()
        
        # Red = Upper Clothes, Green = Lower Clothes, Blue = Skin/Body
        # We combine Red and Green, completely ignoring the Blue skin channel.
        clothes_only = np.maximum(np.array(r), np.array(g))
        mask = Image.fromarray(clothes_only).convert("L")
    else:
        # Standard extraction for general models (IS-Net, U2Netp, etc.)
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
    
