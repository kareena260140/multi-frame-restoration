import streamlit as st
import torch
import torch.nn as nn
import numpy as np
from PIL import Image

st.set_page_config(page_title="Multi-Frame Restoration System", layout="wide")
st.title("Multi-Frame Restoration System")

# ==========================================================
# INSPECT CHECKPOINT IN STREAMLIT
# ==========================================================
device = "cpu"
ckpt_data = torch.load("variantB_best.pt", map_location=device)
state_dict = ckpt_data.get("state_dict", ckpt_data.get("model_state_dict", ckpt_data))

st.subheader("Checkpoint Inspection Data")
keys_and_shapes = {k: list(v.shape) for k, v in list(state_dict.items())[:10]}
st.json(keys_and_shapes)

first_layer_name = list(state_dict.keys())[0]
first_layer_shape = list(state_dict[first_layer_name].shape)
st.write(f"**First Layer Name:** `{first_layer_name}`")
st.write(f"**First Layer Shape:** `{first_layer_shape}`")
