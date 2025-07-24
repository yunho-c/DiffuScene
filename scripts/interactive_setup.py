# Step 1: Imports and setup
import torch
import numpy as np
import logging
import os

# Import functions from the project
from training_utils import load_config
from scene_synthesis.datasets import filter_function, get_dataset_raw_and_encoded
from scene_synthesis.datasets.threed_future_dataset import ThreedFutureDataset
from scene_synthesis.networks import build_network

# Disable trimesh's logger
logging.getLogger("trimesh").setLevel(logging.ERROR)

# Step 2: Set parameters (REPLACE WITH YOUR PATHS if necessary)
# NOTE: These paths are relative to the `scripts` directory.
config_file = "../config/uncond/diffusion_bedrooms_instancond_lat32_v.yaml"
# IMPORTANT: Set the path to your pre-trained model weights
weight_file = None  # e.g., "../pretrained/bedrooms_uncond.pth"
# IMPORTANT: Set the path to your pickled 3D-FUTURE models
path_to_pickled_3d_futute_models = "../demo/threed_future_models.pkl"
path_to_floor_plan_textures = "../demo/floor_plan_texture_images"

# Step 3: Initialize device, config, datasets, and network
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Running code on", device)

config = load_config(config_file)

# Modify config for evaluation as in the original script
if 'text' in config["data"]["encoding_type"] and 'textfix' not in config["data"]["encoding_type"]:
    config["data"]["encoding_type"] = config["data"]["encoding_type"].replace('text', 'textfix')
if "no_prm" not in config["data"]["encoding_type"]:
    print('NO PERM AUG in test')
    config["data"]["encoding_type"] += "_no_prm"
print('Encoding type:', config["data"]["encoding_type"])

# Load datasets
raw_dataset, dataset = get_dataset_raw_and_encoded(
    config["data"],
    filter_fn=filter_function(
        config["data"],
        split=config["validation"].get("splits", ["test"])
    ),
    split=config["validation"].get("splits", ["test"])
)
objects_dataset = ThreedFutureDataset.from_pickled_dataset(
    path_to_pickled_3d_futute_models
)
print(f"Loaded {len(dataset)} scenes and {len(objects_dataset)} 3D models.")

# Build and load network
network, _, _ = build_network(
    dataset.feature_size, dataset.n_classes,
    config, weight_file, device=device
)
network.eval()

print("\nSetup complete. You can now import these objects in your main script.")
