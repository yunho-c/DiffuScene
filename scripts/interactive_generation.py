import torch
import numpy as np
from utils import floor_plan_from_scene, get_textured_objects_based_on_objfeats, get_textured_objects

def generate_scene(
    network,
    dataset,
    raw_dataset,
    objects_dataset,
    config,
    device,
    scene_idx,
    path_to_floor_plan_textures,
    clip_denoised=False,
    retrive_objfeats=False,
    no_texture=False,
    without_floor=False,
    seed=0
):
    """Generates a single scene based on the provided index and parameters."""
    current_scene = raw_dataset[scene_idx]
    samples = dataset[scene_idx]
    print(f"Using floor plan from scene {current_scene.scene_id} (index {scene_idx})")

    # Get a floor plan
    floor_plan, tr_floor, room_mask = floor_plan_from_scene(
        current_scene, path_to_floor_plan_textures, no_texture=no_texture
    )

    # Generate layout
    bbox_params = network.generate_layout(
            room_mask=room_mask.to(device),
            num_points=config["network"]["sample_num_points"],
            point_dim=config["network"]["point_dim"],
            text=samples.get('description'),
            device=device,
            clip_denoised=clip_denoised,
            batch_seeds=torch.arange(seed, seed + 1),
    )

    boxes = dataset.post_process(bbox_params)
    bbox_params_t = torch.cat([
        boxes["class_labels"],
        boxes["translations"],
        boxes["sizes"],
        boxes["angles"]
    ], dim=-1).cpu().numpy()

    # Retrieve 3D models
    classes = np.array(dataset.class_labels)
    if retrive_objfeats:
        objfeats = boxes["objfeats"].cpu().numpy()
        renderables, trimesh_meshes, _ = get_textured_objects_based_on_objfeats(
            bbox_params_t, objects_dataset, classes, diffusion=True, no_texture=no_texture, query_objfeats=objfeats,
        )
    else:
        renderables, trimesh_meshes, _ = get_textured_objects(
            bbox_params_t, objects_dataset, classes, diffusion=True, no_texture=no_texture
        )

    if not without_floor:
        renderables += floor_plan
        trimesh_meshes += tr_floor

    return renderables, trimesh_meshes
