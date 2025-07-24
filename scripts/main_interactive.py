import numpy as np
from simple_3dviz import Scene
from simple_3dviz.window import show
from simple_3dviz.behaviours.movements import CameraTrajectory
from simple_3dviz.behaviours.trajectory import Circle

# Import the setup and generation functions
from interactive_setup import (
    network, dataset, raw_dataset, objects_dataset, config, device,
    path_to_floor_plan_textures
)
from interactive_generation import generate_scene

def main():
    """Main function to generate and visualize a scene."""
    # --- Parameters for generation ---
    scene_idx_to_generate = 25  # Change this to generate from a different floor plan
    generation_seed = np.random.randint(1000)  # Change this for different random results

    print(f"\n--- Generating scene {scene_idx_to_generate} with seed {generation_seed} ---")
    renderables, _ = generate_scene(
        network=network,
        dataset=dataset,
        raw_dataset=raw_dataset,
        objects_dataset=objects_dataset,
        config=config,
        device=device,
        scene_idx=scene_idx_to_generate,
        path_to_floor_plan_textures=path_to_floor_plan_textures,
        clip_denoised=True,
        seed=generation_seed
    )

    # --- Visualize the generated scene ---
    scene = Scene(size=(768, 768), background=(1, 1, 1, 1))
    scene.add_many(renderables)

    # Use camera settings from the script for a good default view
    scene.camera_position = (-0.1, 1.9, -7.2)
    scene.camera_target = (0, 0, 0)
    scene.up_vector = (0, 1, 0)

    # Optional: Add a rotating camera for better inspection
    scene.add_behaviour(CameraTrajectory(
        Circle(center=(0, 0, 0), point=(0, 2.5, -8), normal=(0, 1, 0)),
        speed=0.005
    ))

    # This will open a window to display the scene
    print("\n--- Showing scene. Close the window to exit. ---")
    show(scene)

if __name__ == "__main__":
    main()
