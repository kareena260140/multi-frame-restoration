import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image
import torch.nn as nn
import torch.nn.functional as F

st.set_page_config(page_title="Multi-Frame Restoration System", layout="wide")
st.title("Multi-Frame Restoration System")

# ==========================================================
# 1. MODEL ARCHITECTURE
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
# 2. CACHED MODEL LOADER
# ==========================================================
@st.cache_resource
def load_model():
    device = "cpu"
    net = VariantANet().to(device)
    ckpt_data = torch.load("variantB_best.pt", map_location=device)
    state_dict = ckpt_data.get("state_dict", ckpt_data.get("model_state_dict", ckpt_data))
    net.load_state_dict(state_dict, strict=False)
    net.eval()
    return net, device

model, device = load_model()

# ==========================================================
# 3. RESTORATION INFERENCE LOGIC (Residual + Direct Safe Handling)
# ==========================================================
def restore_image(degraded_ref, prev_frame_1, prev_frame_2, mask_1, mask_2):
    def ensure_rgb(img):
        if img is None:
            return np.zeros((256, 512, 3), dtype=np.uint8)
        img = np.array(img)
        if len(img.shape) == 2:
            return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        if img.shape[2] == 4:
            return cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        return img[:, :, :3]

    def ensure_gray(mask):
        if mask is None:
            return np.zeros((256, 512), dtype=np.uint8)
        mask = np.array(mask)
        if len(mask.shape) == 3:
            return cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        return mask

    orig_h, orig_w = degraded_ref.shape[:2]
    target_w, target_h = 512, 256

    # Ensure strictly clean RGB / Gray channels
    ref_rgb = ensure_rgb(degraded_ref)
    p1_rgb  = ensure_rgb(prev_frame_1)
    p2_rgb  = ensure_rgb(prev_frame_2)
    m1_gray = ensure_gray(mask_1)
    m2_gray = ensure_gray(mask_2)

    # Standardize input dimensions
    ref_res = cv2.resize(ref_rgb, (target_w, target_h), interpolation=cv2.INTER_AREA)
    p1_res  = cv2.resize(p1_rgb,  (target_w, target_h), interpolation=cv2.INTER_AREA)
    p2_res  = cv2.resize(p2_rgb,  (target_w, target_h), interpolation=cv2.INTER_AREA)
    m1_res  = cv2.resize(m1_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
    m2_res  = cv2.resize(m2_gray, (target_w, target_h), interpolation=cv2.INTER_NEAREST)

    # Normalize to [0.0, 1.0] tensors
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

        # Automatic Residual Check:
        # If mean brightness is very low (< 0.15), model predicts a residual map (diff) to add back to reference
        if out.mean().item() < 0.15:
            out = out + t_ref

        out = torch.clamp(out, 0.0, 1.0)

    out_np = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)
    restored_final = cv2.resize(out_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return restored_final

# ==========================================================
# 4. STREAMLIT UI LAYOUT
# ==========================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Frames & Masks")
    f_ref = st.file_uploader("Degraded Reference Frame", type=["png", "jpg", "jpeg"])
    f_p1  = st.file_uploader("Previous Frame 1", type=["png", "jpg", "jpeg"])
    f_p2  = st.file_uploader("Previous Frame 2", type=["png", "jpg", "jpeg"])
    f_m1  = st.file_uploader("Mask 1 (Grayscale)", type=["png", "jpg", "jpeg"])
    f_m2  = st.file_uploader("Mask 2 (Grayscale)", type=["png", "jpg", "jpeg"])
    submit_btn = st.button("Restore Frame", type="primary", use_container_width=True)

with col2:
    st.subheader("Restoration Output")
    if submit_btn:
        if f_ref and f_p1 and f_p2:
            with st.spinner("Processing restoration..."):
                img_ref = np.array(Image.open(f_ref))
                img_p1  = np.array(Image.open(f_p1))
                img_p2  = np.array(Image.open(f_p2))
                img_m1  = np.array(Image.open(f_m1)) if f_m1 else None
                img_m2  = np.array(Image.open(f_m2)) if f_m2 else None

                restored = restore_image(img_ref, img_p1, img_p2, img_m1, img_m2)
                st.image(restored, caption="Restored Output", use_container_width=True)
                st.success("Restoration complete!")
        else:
            st.warning("Please upload at least the Reference Frame, Previous Frame 1, and Previous Frame 2.")
