# ==========================================================
# 3. MEMORY-OPTIMIZED PREDICTION FUNCTION
# ==========================================================
def restore_image(degraded_ref, prev_frame_1, prev_frame_2, mask_1, mask_2):
    if degraded_ref is None or prev_frame_1 is None or prev_frame_2 is None:
        return None

    # Store original dimensions to upscale back at the end
    orig_h, orig_w = degraded_ref.shape[:2]

    # Target size for memory efficiency on Render Free Tier (512 MB RAM)
    target_w, target_h = 512, 256

    # Resize all inputs to target size
    ref_res  = cv2.resize(degraded_ref, (target_w, target_h))
    p1_res   = cv2.resize(prev_frame_1, (target_w, target_h))
    p2_res   = cv2.resize(prev_frame_2, (target_w, target_h))
    
    # Ensure masks are 2D single-channel arrays
    if mask_1 is not None:
        m1 = mask_1 if len(mask_1.shape) == 2 else mask_1[:, :, 0]
        m1_res = cv2.resize(m1, (target_w, target_h))
    else:
        m1_res = np.zeros((target_h, target_w), dtype=np.uint8)

    if mask_2 is not None:
        m2 = mask_2 if len(mask_2.shape) == 2 else mask_2[:, :, 0]
        m2_res = cv2.resize(m2, (target_w, target_h))
    else:
        m2_res = np.zeros((target_h, target_w), dtype=np.uint8)

    # Convert to normalized tensors
    to_t = lambda arr: torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)
    to_m = lambda arr: torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().div(255.0).to(device)

    t_ref = to_t(ref_res)
    t_p1  = to_t(p1_res)
    t_p2  = to_t(p2_res)
    t_m1  = to_m(m1_res)
    t_m2  = to_m(m2_res)

    inp = torch.cat([t_ref, t_p1, t_p2, t_m1, t_m2], dim=1)

    with torch.no_grad():
        out = model(inp)
        if isinstance(out, tuple):
            out = out[0]

    out_np = (out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

    # Resize restored output back to original uploaded frame resolution
    restored_final = cv2.resize(out_np, (orig_w, orig_h))
    return restored_final
