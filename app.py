import gradio as gr
import torch, cv2, numpy as np
import torch.nn as nn

# --- 1. PASTE YOUR VariantANet CLASS HERE ---
# (Copy and paste your 'class VariantANet(nn.Module):' definition from your Colab notebook here)


# --- 2. LOAD MODEL ---
device = "cpu"
model = VariantANet().to(device)

ckpt_data = torch.load("variantB_best.pt", map_location=device)
state_dict = ckpt_data.get("state_dict", ckpt_data.get("model_state_dict", ckpt_data))
model.load_state_dict(state_dict, strict=False)
model.eval()

# --- 3. PREDICT FUNCTION ---
def restore_image(degraded_ref, prev_frame_1, prev_frame_2, mask_1, mask_2):
    to_t = lambda arr: torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)
    to_m = lambda arr: torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().div(255.0).to(device)
    
    t_ref = to_t(degraded_ref)
    t_p1  = to_t(prev_frame_1)
    t_p2  = to_t(prev_frame_2)
    t_m1  = to_m(mask_1)
    t_m2  = to_m(mask_2)
    
    inp = torch.cat([t_ref, t_p1, t_p2, t_m1, t_m2], dim=1)
    
    with torch.no_grad():
        out = model(inp)
        if isinstance(out, tuple):
            out = out[0]
            
    out_np = (out[0].permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).astype(np.uint8)
    return out_np

# --- 4. LAUNCH UI ---
ui = gr.Interface(
    fn=restore_image,
    inputs=[
        gr.Image(label="Degraded Reference Frame"),
        gr.Image(label="Previous Frame 1"),
        gr.Image(label="Previous Frame 2"),
        gr.Image(image_mode="L", label="Mask 1 (Grayscale)"),
        gr.Image(image_mode="L", label="Mask 2 (Grayscale)")
    ],
    outputs=gr.Image(label="Restored Image"),
    title="Multi-Frame Restoration System"
)

# Render expects the service on port 10000
ui.launch(server_name="0.0.0.0", server_port=10000)
