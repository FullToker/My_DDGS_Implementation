<h1 align="center">D<sup>2</sup>GS: Depth-and-Density Guided Gaussian Splatting for Stable and Accurate Sparse-View Reconstruction</h1>

<p align="center"><img src='figures/pipeline.png'></p>

## Overview 

The proposed **D²GS** mainly consists of two key components: a Depth-and-Density guided Dropout (DD-Drop) mechanism and Distance-Aware Fidelity Enhancement (DAFE), to improve the stability and spatial completeness of scene reconstruction under sparse-view settings. DD-Drop assigns each Gaussian a dropout score based on local density and camera distance, indicating regions prone to overfitting. High-scoring Gaussians would be dropped with a higher probability to suppress aliasing and improve rendering fidelity. In addition, DAFE avoids underfitting by boosting supervision in distant regions using depth priors.

## Implementation
### Installation
We provide an installation using Conda package and environment management:
```
git clone https://github.com/Willacold/DDGS
cd DDGS
conda env create --file environment.yaml
conda activate DDGS
```

**Note:** This Conda environment assumes that **CUDA 12.1** is already installed on your system.

### Data Preparation

In the data preparation stage, we first reconstruct sparse-view inputs using **Structure-from-Motion (SfM)** with the provided camera poses from the datasets. Then, we perform dense stereo matching using COLMAP’s `patch_match_stereo` function, followed by `stereo_fusion` to generate the dense stereo point cloud.

```bash
mkdir dataset
cd dataset

# Download LLFF dataset
gdown 16VnMcF1KJYxN9QId6TClMsZRahHNMW5g

# Generate sparse point cloud using COLMAP (limited views) for LLFF
python tools/colmap_llff.py

# Download MipNeRF-360 dataset
wget http://storage.googleapis.com/gresearch/refraw360/360_v2.zip
unzip -d mipnerf360 360_v2.zip

# Generate sparse point cloud using COLMAP (limited views) for MipNeRF-360
python tools/colmap_360.py
```


### Training

#### LLFF Dataset

To train on a single LLFF scene, use the following command:

```
python train.py -s dataset/nerf_llff_data/trex -m output_llff --depth_weight 0.15 --density_weight 0.85 --drop_min 0.05 --drop_max 0.5 --mask_param 5 --lambda_far 0.5 --eval -r 8 --n_views 3
```
To train and evaluate on **all LLFF scenes**, simply run the script below:
```
bash scripts/train_llff.sh
```

#### MipNeRF-360 Dataset

To train on a single MipNeRF-360 scene, use the following command:

```
python train.py -s dataset/mipnerf360/kitchen -m output_mipnerf360 --depth_weight 0.1 --density_weight 0.9 --drop_min 0.05 --drop_max 0.5 --mask_param 10 --lambda_far 0.5 --eval -r 8 --n_views 12
```
To train and evaluate on **all MipNeRF-360 scenes**, simply run the script below:
```
bash scripts/train_mipnerf360.sh
```

### Rendering & Evaluation
You can perform **rendering and evaluation in a single step** using the following command:
#### LLFF Dataset
```
python render.py -s -m output_llff --eval -r 8
```
#### MipNeRF-360 Dataset
```
python render.py -s -m output_mipnerf360 --eval -r 8
```

## Acknowledgement
Our implementation and experiments are built on top of open-source GitHub repositories. We thank all the authors who made their code public, which tremendously accelerates our project progress. If you find these works helpful, please consider citing them as well.

[DCVL-3D/DropGaussian_release](https://github.com/DCVL-3D/DropGaussian_release)  </br>
[GraphDeco-INRIA/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting)  </br>

<!-- ## Citation
If you find our work useful for your project, please consider citing the following paper.
```

``` -->
