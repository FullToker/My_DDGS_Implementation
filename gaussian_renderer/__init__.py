#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, \
           override_color = None, is_train=False, iteration=None, depth_weight=0.5, density_weight=0.5, drop_min=0.05, drop_max=0.3):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None

    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color
    

    # Apply DDGS
    if is_train:
        gaussian_positions = pc.get_xyz  
        # camera_depths = torch.norm(gaussian_positions - viewpoint_camera.camera_center, dim=1) + 1e-6 
        ones = torch.ones((gaussian_positions.shape[0], 1), device=gaussian_positions.device)
        gaussian_positions_homo = torch.cat([gaussian_positions, ones], dim=1)  # [N, 4]
        camera_coordinates = torch.matmul(gaussian_positions_homo, viewpoint_camera.world_view_transform.T)  # [N, 4]
        camera_depths = camera_coordinates[:, 2] 
        depth_min, depth_max = camera_depths.min(), camera_depths.max()
        depth_score = (1.0 - (camera_depths - depth_min) / (depth_max - depth_min + 1e-6)).float() 
        
        sorted_depths, _ = torch.sort(camera_depths)
        n = sorted_depths.shape[0]
        idx_33 = int(n * 0.33)
        idx_67 = int(n * 0.67)
        depth_percentile_33 = sorted_depths[idx_33].float()
        depth_percentile_67 = sorted_depths[idx_67].float()
        
        near_field = camera_depths <= depth_percentile_33
        mid_field = (camera_depths > depth_percentile_33) & (camera_depths <= depth_percentile_67)
        far_field = camera_depths > depth_percentile_67
        
        density_norm = torch.ones_like(depth_score) * 0.5  
        if pc.density_score.numel() >= opacity.shape[0]:
            density_score = (pc.density_score[:opacity.shape[0]] + 1e-6).float()
            density_norm = ((density_score - density_score.min()) / (density_score.max() - density_score.min() + 1e-6)).float()      
            
        combined_score = (depth_weight * depth_score + density_weight * density_norm).float()
        
        progress = min(1.0, iteration / 10000.0)
        drop_rate = float(drop_min + (drop_max - drop_min) * progress)
        
        drop_prob = (near_field.float() * combined_score * drop_rate + 
                    mid_field.float() * combined_score * drop_rate * 0.7 + 
                    far_field.float() * combined_score * drop_rate * 0.3)
        
        keep_prob = 1.0 - drop_prob
        mask = (torch.rand_like(keep_prob) < keep_prob).float()
        opacity = opacity * mask[:, None]


    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    rendered_image = rendered_image.clamp(0, 1)
    out = {
        "render": rendered_image,
        "viewspace_points": screenspace_points,
        "visibility_filter" : (radii > 0),
        "radii": radii
        }
    
    return out
