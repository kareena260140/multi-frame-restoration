import gradio as gr
import torch
import cv2
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

# ==========================================================
# 1. MODEL ARCHITECTURE (Matches variantB_best.pt checkpoint)
# ==========================================================
class VariantANet(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )

        self.enc1 = block(11, 32)
        self.enc2 = block(32, 64)
        self.enc3 = block(64, 128)
        self.pool = nn.MaxPool2d(2)

        self.bottleneck = block(128, 256)

        # Upsampling and decoder blocks matching saved weights:
        # 256 + 128 = 384, 128 + 64 = 192, 64 + 32 = 96
        self.up3 = nn.ConvTranspose2d(256, 256, 2, stride=2)
        self.dec3 = block(384, 128)

        self.up2 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec2 = block(192, 64)

        self.up1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec1 = block(96, 32)

        self.final = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b)
        if d3.shape[2:] != e3.shape[2:]:
            d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        if d2.shape[2:] != e2.shape[2:]:
            d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        if d1.shape[2:] != e1.shape[2:]:
            d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final(d1)
        return out


# ==========================================================
# 2. LOAD MODEL & CHECKPOINT WEIGHTS
# ==========================================================
device = "cpu"
model = VariantANet().to(device)

ckpt_data = torch.load("variantB_best.pt", map_location=device)
state_dict = ckpt_data.get("state_dict", ckpt_data.get("model_state_dict", ckpt_data))
model.load_state_dict(state_dict, strict=False)
model.eval()


# ==========================================================
# 3. ROBUST PREDICTION FUNCTION (Safe Resizing + Channel Guard)
# ==========================================================
def restore_image(degraded_ref, prev_frame_1, prev_frame_2, mask_1, mask_2):
    if degraded_ref is None or prev_frame_1 is None or prev_frame_2 is None:
        return None

    # Helper: Ensure strict 3-channel RGB
    def ensure_rgb(img):
        if img is None:
            return np.zeros((256, 512, 3), dtype=np.uint8)
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img[:, :, :3]

    # Helper: Ensure strict 1-channel Grayscale
    def ensure_gray(mask):
        if mask is None:
            return np.zeros((256, 512), dtype=np.uint8)
        if len(mask.shape) == 3:
            return cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        return mask

    orig_h, orig_w = degraded_ref.shape[:2]
    target_w, target_h = 512, 256

    # 1. Clean channels
    ref_rgb = ensure_rgb(degraded_ref)
    p1_rgb  = ensure_rgb(prev_frame_1)
    p2_rgb  = ensure_rgb(prev_frame_2)
    m1_gray = ensure_gray(mask_1)
    m2_gray = ensure_gray(mask_2)

    # 2. Resize to 512x256 to fit Render 512MB RAM budget
    ref_res = cv2.resize(ref_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    p1_res  = cv2.resize(p1_rgb,  (target_w, target_h), interpolation=cv2.INTER_AREA)
    p2_res  = cv2.resize(p2_rgb,  (target_w, target_h), interpolation=cv2.INTER_AREA)
    m1_res  = cv2.resize(m1_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    m2_res  = cv2.resize(m2_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # 3. Convert to Torch Tensors
    to_t = lambda arr: torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float().div(255.0).to(device)
    to_m = lambda arr: torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float().div(255.0).to(device)

    t_ref = to_t(ref_res)
    t_p1  = to_t(p1_res)
    t_p2  = to_t(p2_res)
    t_m1  = to_m(m1_res)
    t_m2  = to_m(m2_res)

    # Exactly 3 + 3 + 3 + 1 + 1 = 11 channels
    inp = torch.cat([t_ref, t_p1, t_p2, t_m1, t_m2], dim=1)

    with torch.no_grad():
        out = model(inp)
        if isinstance(out, tuple):
            out = out[0]

    out_np = (out[0].permute(1, 2, 0).clamp(0, 1).cpu().numpy() * 255.0).astype(np.uint8)

    # Resize back to original frame dimensions
    restored_final = cv2.resize(out_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return restored_final


# ==========================================================
# 4. LAUNCH UI ON PORT 10000 FOR RENDER
# ==========================================================
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

ui.launch(server_name="0.0.0.0", server_port=10000)
