import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

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

        return self.final(d1)

# ==========================================================
# 2. CACHED MODEL LOADER
# ==========================================================
@st.cache_resource
def load_model():
    device = "cpu"
    net = VariantANet().to(device)
    ckpt_data = torch.load("variantB_best.pt", map_location=device)
    state_dict = ckpt_data.get("state_dict", ckpt_data.get("model_state_dict", ckpt_data))

    # Clean prefixes if saved under DataParallel or container
    cleaned_dict = {}
    for k, v in state_dict.items():
        name = k.replace("module.", "").replace("model.", "")
        cleaned_dict[name] = v

    net.load_state_dict(cleaned_dict, strict=False)
    net.eval()
    return net, device

model, device = load_model()

# ==========================================================
# 3. RESTORATION INFERENCE LOGIC
# ==========================================================
def restore_image(pil_ref, pil_p1, pil_p2, pil_m1, pil_m2, channel_mode, frame_order, add_res):
    orig_w, orig_h = pil_ref.size

    # Convert PIL to pure normalized float arrays [0.0, 1.0]
    ref_rgb = np.array(pil_ref.convert("RGB"), dtype=np.float32) / 255.0
    p1_rgb  = np.array(pil_p1.convert("RGB").resize((orig_w, orig_h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0
    p2_rgb  = np.array(pil_p2.convert("RGB").resize((orig_w, orig_h), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0

    if pil_m1 is not None:
        m1_gray = np.array(pil_m1.convert("L").resize((orig_w, orig_h), Image.Resampling.NEAREST), dtype=np.float32) / 255.0
    else:
        m1_gray = np.zeros((orig_h, orig_w), dtype=np.float32)

    if pil_m2 is not None:
        m2_gray = np.array(pil_m2.convert("L").resize((orig_w, orig_h), Image.Resampling.NEAREST), dtype=np.float32) / 255.0
    else:
        m2_gray = np.zeros((orig_h, orig_w), dtype=np.float32)

    to_t = lambda arr: torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).to(device)
    to_m = lambda arr: torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).to(device)

    t_ref = to_t(ref_rgb)
    t_p1  = to_t(p1_rgb)
    t_p2  = to_t(p2_rgb)
    t_m1  = to_m(m1_gray)
    t_m2  = to_m(m2_gray)

    # Input frame ordering selection
    if frame_order == "[t-2, t-1, t] (Sequential)":
        inp = torch.cat([t_p2, t_p1, t_ref, t_m2, t_m1], dim=1)
    else:
        inp = torch.cat([t_ref, t_p1, t_p2, t_m1, t_m2], dim=1)

    # Pad to multiples of 8
    pad_h = (8 - orig_h % 8) % 8
    pad_w = (8 - orig_w % 8) % 8
    if pad_h > 0 or pad_w > 0:
        inp = F.pad(inp, (0, pad_w, 0, pad_h), mode='replicate')

    with torch.no_grad():
        out = model(inp)
        if isinstance(out, tuple):
            out = out[0]

        out = out[:, :, :orig_h, :orig_w]

        # Add reference frame if model predicts residual delta
        if add_res:
            out = out + t_ref

        out = out.clamp(0.0, 1.0)

    # Extract [H, W, 3] float array
    out_arr = out[0].permute(1, 2, 0).cpu().numpy()

    # Re-align channels according to selected mapping
    if channel_mode == "Swap Red & Green (Fix Magenta)":
        # Swap Channel 0 and 1: [G, R, B] -> [R, G, B]
        out_arr = out_arr[:, :, [1, 0, 2]]
    elif channel_mode == "Swap Red & Blue (BGR)":
        # Swap Channel 0 and 2: [B, G, R] -> [R, G, B]
        out_arr = out_arr[:, :, [2, 1, 0]]
    elif channel_mode == "Swap Green & Blue (RBG)":
        out_arr = out_arr[:, :, [0, 2, 1]]

    out_uint8 = (out_arr * 255.0).round().astype(np.uint8)
    return Image.fromarray(out_uint8, mode="RGB")

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

    st.markdown("---")
    st.subheader("Color & Channel Alignment")
    channel_mode = st.selectbox(
        "Channel Mapping:",
        [
            "Swap Red & Green (Fix Magenta)",
            "RGB (Direct)",
            "Swap Red & Blue (BGR)",
            "Swap Green & Blue (RBG)"
        ],
        index=0
    )
    frame_order = st.radio(
        "Frame Concatenation Order:",
        ["[t-2, t-1, t] (Sequential)", "[t, t-1, t-2] (Reference First)"],
        index=0
    )
    add_residual = st.checkbox("Apply Residual Addition (+ Reference Frame)", value=True)
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

                restored = restore_image(
                    img_ref, img_p1, img_p2, img_m1, img_m2,
                    channel_mode=channel_mode,
                    frame_order=frame_order,
                    add_res=add_residual
                )
                st.image(restored, caption=f"Restored Output ({channel_mode})", use_container_width=True)
                st.success("Restoration complete!")
        else:
            st.warning("Please upload Reference Frame (t), Previous Frame 1 (t-1), and Previous Frame 2 (t-2).")
