# DiffuScene System Diagram

This document outlines the architecture and data flow of the DiffuScene project.

## System Overview

The DiffuScene system is designed to generate 3D scenes using a diffusion model. The workflow is divided into three main stages: Data Preparation, Training, and Inference.

1.  **Data Preparation**: Raw 3D scene data from the 3D-FRONT and 3D-FUTURE datasets is processed by scripts in the `scripts/` directory. The core logic in `scene_synthesis/datasets/` handles filtering, splitting (using `.csv` files from `config/`), and encoding the scenes into a vectorized format (positions, sizes, classes, etc.). This processed data is cached for efficient use.

2.  **Training**: The training process is initiated by scripts like `scripts/train_diffusion.py`. These scripts use `.yaml` configuration files from the `config/` directory to define the model architecture, dataset, and hyperparameters. The central model, `DiffusionSceneLayout_DDPM`, uses a `Unet1D` as its denoising backbone. The model can be trained unconditionally or conditioned on inputs like text descriptions or partial scenes.

3.  **Inference (Generation)**: Scene generation is performed by scripts like `scripts/generate_diffusion.py`, which load a pre-trained model. The model generates a scene layout (object parameters like class, position, size). Utility functions in `scene_synthesis/utils.py` then retrieve corresponding 3D models from the dataset to construct the final 3D scene. The `demo/` directory contains example assets and outputs.

## Mermaid Diagram

```mermaid
graph TD
    subgraph "Data Sources"
        A1[3D-FRONT Dataset]
        A2[3D-FUTURE Models]
    end

    subgraph "Data Preparation Pipeline"
        B1(scripts/preprocess_data.py)
        B2(scene_synthesis/datasets/*)
        B3[Cached & Processed Dataset]

        A1 --> B1
        A2 --> B1
        B1 --> B2
        B2 --> B3
    end

    subgraph "Configuration"
        C1[config/**/*.yaml]
        C1 -- defines paths & hyperparameters --> B2
        C1 -- defines model architecture & training params --> D1
        C1 -- defines model path & generation params --> E1
    end

    subgraph "Training Pipeline"
        D1(scripts/train_diffusion.py)
        D2(scene_synthesis/networks/DiffusionSceneLayout_DDPM)
        D3(scene_synthesis/networks/denoise_net.py - Unet1D)
        D4(scene_synthesis/networks/diffusion_ddpm.py)
        D5[Trained Model Checkpoint]

        B3 --> D1
        D1 -- instantiates --> D2
        D2 -- uses --> D3
        D2 -- uses --> D4
        D2 --> D5
    end

    subgraph "Inference Pipeline"
        E1(scripts/generate_diffusion.py)
        E2(scene_synthesis/utils.py)
        E3[Generated 3D Scene]
        E4(demo/*)

        D5 --> E1
        B3 -- for object retrieval --> E1
        E1 -- generates layout --> E2
        E2 -- retrieves models & places them --> E3
        E3 --> E4
    end

    subgraph "Conditional Inputs (Optional)"
        F1[Text Descriptions]
        F2[Partial Scenes for Completion/Rearrangement]
        F1 --> D1
        F2 --> D1
        F1 --> E1
        F2 --> E1
    end

    style B3 fill:#f9f,stroke:#333,stroke-width:2px
    style D5 fill:#f9f,stroke:#333,stroke-width:2px
    style E3 fill:#ccf,stroke:#333,stroke-width:2px
    style A1 fill:#dfd,stroke:#333,stroke-width:2px
    style A2 fill:#dfd,stroke:#333,stroke-width:2px
```
