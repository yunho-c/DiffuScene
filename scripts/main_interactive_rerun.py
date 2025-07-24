import numpy as np
import rerun as rr
import trimesh

# Import the setup and generation functions
from interactive_setup import (
    network,
    dataset,
    raw_dataset,
    objects_dataset,
    config,
    device,
    path_to_floor_plan_textures,
)
from interactive_generation import generate_scene


def main():
    """Main function to generate and visualize a scene with rerun."""
    # --- Parameters for generation ---
    # scene_idx_to_generate = 25  # Change this to generate from a different floor plan
    scene_idx_to_generate = 42  # Change this to generate from a different floor plan
    generation_seed = np.random.randint(
        1000
    )  # Change this for different random results
    # text_prompt = "a room with a bed and a chair"  # Change this to your desired text prompt, or set to None to use the dataset's description
    # text_prompt = "The room has a dining chair, an armchair and a corner side table. There is a second corner side table to the right of the armchair. There is a multi seat sofa in front of the first corner side table."
    text_prompt = "The room has a bed, a dressing table, a lamp, and a small couch under the bed."

    rr.init("DiffuScene")
    rr.serve_web()
    # # Start a gRPC server and use it as log sink.
    # server_uri = rr.serve_grpc()
    # # Connect the web viewer to the gRPC server and open it in the browser
    # rr.serve_web_viewer(connect_to=server_uri)

    print(
        f"--- Generating scene {scene_idx_to_generate} with seed {generation_seed} ---"
    )
    _, trimesh_meshes = generate_scene(
        network=network,
        dataset=dataset,
        raw_dataset=raw_dataset,
        objects_dataset=objects_dataset,
        config=config,
        device=device,
        scene_idx=scene_idx_to_generate,
        path_to_floor_plan_textures=path_to_floor_plan_textures,
        clip_denoised=True,
        seed=generation_seed,
        text_prompt=text_prompt,
    )

    # --- Visualize the generated scene ---
    for i, mesh in enumerate(trimesh_meshes):
        # If the mesh has a material with an image, use it as the texture
        # actually, vertex colors
        if hasattr(mesh.visual, "vertex_colors"):
            # (isinstance(mesh.visual, trimesh.visual.texture.TextureVisuals)
            # and mesh.visual.material.image is not None)

            # Convert PIL image to numpy array
            texture = np.array(mesh.visual.material.image)
            # Log the mesh with texture
            rr.log(
                f"trimesh_{i}",
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    vertex_normals=mesh.vertex_normals,
                    vertex_colors=mesh.visual.vertex_colors[:, :3],
                    triangle_indices=mesh.faces,
                    # mesh_material=rr.Material(albedo_texture=texture)
                ),
            )
        else:
            # Log the mesh with no colors
            rr.log(
                f"trimesh_{i}",
                rr.Mesh3D(
                    vertex_positions=mesh.vertices,
                    vertex_normals=mesh.vertex_normals,
                    triangle_indices=mesh.faces,
                ),
            )


if __name__ == "__main__":
    main()
