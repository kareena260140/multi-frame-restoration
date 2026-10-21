import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image

st.set_page_config(page_title="Multi-Frame Restoration System", layout="wide")
st.title("Multi-Frame Restoration System")

# ==========================================================
# 1. EXACT ARCHITECTURE MATCHING CHECKPOINT KEYS
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

        # Bilinear upsampling is parameterless, matching the checkpoint
        self.dec3 = block(384, 128)
        self.dec2 = block(192, 64)
        self.dec1 = block(96, 32)

        # Exact key matching: 'out_head'
        self.out_head = nn.Conv2d(32, 3, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = F.interpolate(b, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = F.interpolate(d3, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = F.interpolate(d2, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        return self.out_head(d1)

# ==========================================================
# 2. CACHED MODEL LOADER (Strictly verified)
# ==========================================================
@st.cache_resource
def load_model():
    device = "cpu"
    net = VariantANet().to(device)
    ckpt = torch.load("variantB_best.pt", map_location=device)
    state_dict = ckpt.get("state_dict", ckpt.get("model_state_dict", ckpt))

    # Strip prefixes if present
    cleaned_dict = {k.replace("module.", "").replace("model.", ""): v for k, v in state_dict.items()}

    # All keys now match strictly
    net.load_state_dict(cleaned_dict, strict=True)
    net.eval()
    return net, device

model, device = load_model()

# ==========================================================
# 3. INFERENCE PIPELINE
# ==========================================================
def restore_image(pil_ref, pil_p1, pil_p2, pil_m1, pil_m2):
    orig_w, orig_h = pil_ref.size

    # Load images in standard RGB [0.0, 1.0]
    ref_np = np.array(pil_ref.convert("RGB"), dtype=np.float32) / 255.0
    p1_np  = np.array(pil_p1.convert("RGB").resize((orig_w, orig_h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    p2_np  = np.array(pil_p2.convert("RGB").resize((orig_w, orig_h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

    if pil_m1 is not None:
        m1_np = np.array(pil_m1.convert("L").resize((orig_w, orig_h), Image.Resampling.NEAREST), dtype=np.float32) / 255.0
    else:
        m1_np = np.zeros((orig_h, orig_w), dtype=np.float32)

    if pil_m2 is not None:
        m2_np = np.array(pil_m2.convert("L").resize((orig_w, orig_h), Image.Resampling.NEAREST), dtype=np.float32) / 255.0
    else:
        m2_np = np.zeros((orig_h, orig_w), dtype=np.float32)

    to_t = lambda arr: torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    to_m = lambda arr: torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    t_ref = to_t(ref_np)
    t_p1  = to_t(p1_np)
    t_p2  = to_t(p2_np)
    t_m1  = to_m(m1_np)
    t_m2  = to_m(m2_np)

    # Input channels: [t-2, t-1, t, mask2, mask1]
    inp = torch.cat([t_p2, t_p1, t_ref, t_m2, t_m1], dim=1)

    pad_h = (8 - orig_h % 8) % 8
    pad_w = (8 - orig_w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        inp = F.pad(inp, (0, pad_w, 0, pad_h), mode='replicate')

    with torch.no_grad():
        out = model(inp)
        out = out[:, :, :orig_h, :orig_w]

        # Residual check: if output represents delta, add reference frame
        if out.mean().item() < 0.20:
            out = out + t_ref

        out = out.clamp(0.0, 1.0)

    out_np = (out[0].permute(1, 2, 0).cpu().numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(out_np, mode="RGB")

# ==========================================================
# 4. STREAMLIT UI
# ==========================================================
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input Frames & Masks")
    f_ref = st.file_uploader("Degraded Reference Frame (t)", type=["png", "jpg", "jpeg"])
    f_p1  = st.file_uploader("Previous Frame 1 (t-1)", type=["png", "jpg", "jpeg"])
    f_p2  = st.file_uploader("Previous Frame 2 (t-2)", type=["png", "jpg", "jpeg"])
    f_m1  = st.file_uploader("Mask 1 (t-1 Grayscale)", type=["png", "jpg", "jpeg"])
    f_m2  = st.file_uploader("Mask 2 (t-2 Grayscale)", type=["png", "jpg", "jpeg"])

    submit_btn = st.button("Restore Frame", type="primary", use_container_width=True)

with col2:
    st.subheader("Restoration Output")
    if submit_btn:
        if f_ref and f_p1 and f_p2:
            with st.spinner("Processing restoration..."):
                img_ref = Image.open(f_ref)
                img_p1  = Image.open(f_p1)
                img_p2  = Image.open(f_p2)
                img_m1  = Image.open(f_m1) if f_m1 else None
                img_m2  = Image.open(f_m2) if f_m2 else None

                restored = restore_image(img_ref, img_p1, img_p2, img_m1, img_m2)
                st.image(restored, caption="Restored Output", use_container_width=True)
                st.success("Restoration complete!")
        else:
            st.warning("Please upload Reference Frame (t), Previous Frame 1 (t-1), and Previous Frame 2 (t-2).")
