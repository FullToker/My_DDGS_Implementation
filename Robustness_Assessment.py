import time
import numpy as np
import torch
from typing import List, Dict, Optional
from pykeops.torch import LazyTensor
from scene.gaussian_model import GaussianModel
import json
import os
import glob
import argparse
import csv
from scene.cameras import Camera

class MW2StabilityMetric:
    def __init__(self, 
                 epsilon: float = 0.1,          # 1e-2
                 max_iter: int = 10000,          
                 tolerance: float = 1e-4,        
                 max_gaussians: int = 10000,     
                 device: str = 'cuda'): 
        
        if device == 'cuda' and not torch.cuda.is_available():
            device = 'cpu'
        
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.max_gaussians = max_gaussians
        self.device = device
    
    def extract_gaussian_params(self, model_path: str) -> Dict[str, torch.Tensor]:
        model = GaussianModel(3)
        model.load_ply(model_path)
        
        with torch.no_grad():
            means = model.get_xyz           
            scales = model.get_scaling      
            rotations = model.get_rotation  
            opacities = model.get_opacity  
            indices = self._sample_center_depth_range(means, model_path)
            means = means[indices]
            scales = scales[indices]
            rotations = rotations[indices]
            opacities = opacities[indices]
            
            covariances = self._build_covariance_matrices(scales, rotations)
            
            weights = opacities.squeeze()
            
            return {
                'means': means,
                'covariances': covariances,
                'weights': weights
            }
    
    def _sample_center_depth_range(self, means: torch.Tensor, model_path: str) -> torch.Tensor:
        model_dir = os.path.dirname(model_path)  
        point_cloud_dir = os.path.dirname(model_dir)  
        run_dir = os.path.dirname(point_cloud_dir)  
        cameras_json_path = os.path.join(run_dir, "cameras.json")
        
        if not os.path.exists(cameras_json_path):
            raise FileNotFoundError(f"cameras.json not found at {cameras_json_path}")
        
        with open(cameras_json_path, 'r') as f:
            cameras_data = json.load(f)
        camera_info = cameras_data[0]

        R = np.array(camera_info['rotation'])
        T = np.array(camera_info['position'])
        FoVx = camera_info['fx']  
        FoVy = camera_info['fy'] 
        width = camera_info['width']
        height = camera_info['height']
        
        dummy_image = torch.zeros((3, height, width), dtype=torch.float32)
        
        camera = Camera(
            colmap_id=camera_info['id'],
            R=R, T=T, FoVx=FoVx, FoVy=FoVy,
            image=dummy_image, gt_alpha_mask=None,
            image_name=camera_info['img_name'], uid=0,
            bounds=None, trans=np.array([0.0, 0.0, 0.0]), scale=1.0
        )
        
        gaussian_positions = means
        ones = torch.ones((gaussian_positions.shape[0], 1), device=gaussian_positions.device)
        gaussian_positions_homo = torch.cat([gaussian_positions, ones], dim=1)
        camera_coordinates_homo = torch.matmul(gaussian_positions_homo, camera.world_view_transform.T)
        camera_coordinates = camera_coordinates_homo[:, :3]
        camera_depths = camera_coordinates[:, 2]
        camera_depths = torch.abs(camera_depths) + 1e-6
        depths = camera_depths
    
        difference = 2
        n_gaussians = means.shape[0]
        n_samples = min(self.max_gaussians, n_gaussians)
        sorted_indices = torch.argsort(depths)
        base_step = n_gaussians // n_samples
        
        far_step = max(1, base_step + difference)
        far_start = 0
        far_end = n_gaussians // 3
        far_indices = sorted_indices[far_start:far_end:far_step]
        
        mid_step = base_step
        mid_start = n_gaussians // 3
        mid_end = 2 * n_gaussians // 3
        mid_indices = sorted_indices[mid_start:mid_end:mid_step]

        near_step = max(1, base_step - difference)
        near_start = 2 * n_gaussians // 3
        near_end = n_gaussians
        near_indices = sorted_indices[near_start:near_end:near_step]

        selected_indices = torch.cat([far_indices, mid_indices, near_indices])
    
        return selected_indices
    
    def _build_covariance_matrices(self, scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
        K = scales.shape[0]
        L = self._build_scaling_rotation(scales, rotations)  
        covariances = torch.bmm(L, L.transpose(-2, -1)) 
        return covariances
    
    def _build_scaling_rotation(self, scales: torch.Tensor, rotations: torch.Tensor) -> torch.Tensor:
        K = scales.shape[0]
        L = torch.zeros((K, 3, 3), dtype=scales.dtype, device=scales.device)
        R = self._quaternion_to_rotation_matrix(rotations)
        L[:, 0, 0] = scales[:, 0]
        L[:, 1, 1] = scales[:, 1]
        L[:, 2, 2] = scales[:, 2]
        L = torch.bmm(R, L)
        return L
    
    def _quaternion_to_rotation_matrix(self, q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
        norm = torch.sqrt(w**2 + x**2 + y**2 + z**2)
        w, x, y, z = w/norm, x/norm, y/norm, z/norm
        R = torch.zeros(q.shape[0], 3, 3, device=q.device, dtype=q.dtype)
        R[:, 0, 0] = 1 - 2*(y**2 + z**2)
        R[:, 0, 1] = 2*(x*y - w*z)
        R[:, 0, 2] = 2*(x*z + w*y)
        R[:, 1, 0] = 2*(x*y + w*z)
        R[:, 1, 1] = 1 - 2*(x**2 + z**2)
        R[:, 1, 2] = 2*(y*z - w*x)
        R[:, 2, 0] = 2*(x*z - w*y)
        R[:, 2, 1] = 2*(y*z + w*x)
        R[:, 2, 2] = 1 - 2*(x**2 + y**2)
        return R
    
    def sanitize_covs(self, covs, eps_scale=1e-3):
        covs = 0.5 * (covs + covs.transpose(-1, -2))
        eigvals, eigvecs = torch.linalg.eigh(covs)
        scene_scale = covs.diagonal(dim1=-2, dim2=-1).max().sqrt().mean()
        floor = (scene_scale * eps_scale) ** 2         
        eigvals_clamped = eigvals.clamp(min=floor)
        covs_spd = (eigvecs * eigvals_clamped.unsqueeze(-2)) @ eigvecs.transpose(-1, -2)
        return covs_spd          


    def sinkhorn_knopp(self, means1, covs1, w1, means2, covs2, w2):
        covs1 = self.sanitize_covs(covs1)
        covs2 = self.sanitize_covs(covs2)
        means1, covs1, w1 = means1.to(self.device), covs1.to(self.device), w1.to(self.device)
        means2, covs2, w2 = means2.to(self.device), covs2.to(self.device), w2.to(self.device)
        K, L = means1.shape[0], means2.shape[0]

        w1_normalized = w1 / w1.sum()
        w2_normalized = w2 / w2.sum()
        log_w1, log_w2 = (w1_normalized + 1e-12).log(), (w2_normalized + 1e-12).log()

        x_i = LazyTensor(means1[:, None, :])               
        y_j = LazyTensor(means2[None, :, :])            
        position_dist = ((x_i - y_j) ** 2).sum(-1)          

        covs2_inv = torch.linalg.inv(covs2)                
        covs1_f       = covs1.contiguous().view(K, 9)       
        covs2_f       = covs2.contiguous().view(L, 9)       
        covs2_inv_f   = covs2_inv.contiguous().view(L, 9)  
        c_i  = LazyTensor(covs1_f[:, None, :])              
        c_j  = LazyTensor(covs2_f[None, :, :])              
        invj = LazyTensor(covs2_inv_f[None, :, :])         
        delta = c_i - c_j                                  
        d00, d10, d20 = delta[:, :, 0], delta[:, :, 3], delta[:, :, 6]   
        d01, d11, d21 = delta[:, :, 1], delta[:, :, 4], delta[:, :, 7]  
        d02, d12, d22 = delta[:, :, 2], delta[:, :, 5], delta[:, :, 8]   
        C00 = d00*d00 + d10*d10 + d20*d20
        C11 = d01*d01 + d11*d11 + d21*d21
        C22 = d02*d02 + d12*d12 + d22*d22
        C01 = d00*d01 + d10*d11 + d20*d21
        C02 = d00*d02 + d10*d12 + d20*d22
        C12 = d01*d02 + d11*d12 + d21*d22
        a00, a01, a02 = invj[:, :, 0], invj[:, :, 1], invj[:, :, 2]
        a11, a12      = invj[:, :, 4], invj[:, :, 5]           
        a22           = invj[:, :, 8]                         
        shape_term = (a00 * C00 + a11 * C11 + a22 * C22 + 2.0 * (a01 * C01 + a02 * C02 + a12 * C12)) * 0.25                                      
   
        pos_row_max = position_dist.max(dim=1)
        if isinstance(pos_row_max, tuple):
            pos_row_max = pos_row_max[0]
        pos_max = pos_row_max.max().item()
        T = pos_max         
        shape_term = shape_term - (shape_term - T).relu()  
        shape_term = shape_term.relu()  

        D_ij = position_dist + shape_term          

        total_sum = D_ij.sum(dim=1).sum(dim=0)         
        scale = total_sum.item() / (K * L)  
        # scale = 1           
        D_ij_scaled = D_ij / scale                   

        eps_t = self.epsilon     
        log_u = torch.zeros(K, device=self.device)
        log_v = torch.zeros(L, device=self.device)

        best_error = float('inf')
        best_log_u = log_u.clone()
        best_log_v = log_v.clone()
        stagnation_count = 0
        
        for iteration in range(self.max_iter):
            log_u_prev = log_u.clone()

            log_v_j = LazyTensor(log_v[:, None], axis=1)
            cost_term = -D_ij_scaled/eps_t + log_v_j
            logsumexp_result = cost_term.logsumexp(axis=1)
            log_u = log_w1 - logsumexp_result.view(-1)

            log_u_i = LazyTensor(log_u[:, None], axis=0)
            cost_term = -D_ij_scaled/eps_t + log_u_i
            logsumexp_result = cost_term.logsumexp(axis=0)
            log_v = log_w2 - logsumexp_result.view(-1)

            u_change = (log_u - log_u_prev).abs().max().item()
            
            if u_change < best_error:
                best_error = u_change
                best_log_u = log_u.clone()
                best_log_v = log_v.clone()
                stagnation_count = 0
            else:
                stagnation_count += 1
            if u_change < self.tolerance:
                break
            if scale > 1e+03:
                pass
            else:
                if stagnation_count > 4000:
                    log_u = best_log_u
                    log_v = best_log_v
                    break
            if torch.isnan(log_u).any() or torch.isnan(log_v).any():
                log_u = best_log_u
                log_v = best_log_v
                break

        else:
            log_u = best_log_u
            log_v = best_log_v

        u_i = LazyTensor(log_u[:, None], axis=0)
        v_j = LazyTensor(log_v[:, None], axis=1)
        
        exponent_term = u_i + v_j - D_ij_scaled/eps_t
        gamma = exponent_term.exp()
        mw2_sq = (gamma * D_ij_scaled).sum(dim=1).sum(dim=0).item() * scale
        return mw2_sq
    
    
    def compute_mw2_distance(self, model_path1: str, model_path2: str) -> float:
        params1 = self.extract_gaussian_params(model_path1)
        params2 = self.extract_gaussian_params(model_path2)
        means1 = params1['means'].to(self.device)
        covs1 = params1['covariances'].to(self.device)
        weights1 = params1['weights'].to(self.device)
        means2 = params2['means'].to(self.device)
        covs2 = params2['covariances'].to(self.device)
        weights2 = params2['weights'].to(self.device)
        mw2_sq = self.sinkhorn_knopp(means1, covs1, weights1, means2, covs2, weights2)
        mw2_distance = np.sqrt(mw2_sq)
        return mw2_distance
            
    
    def evaluate_training_stability(self, model_paths: List[str]) -> Dict[str, float]:
        n_models = len(model_paths)
        
        distances = []
        distance_matrix = np.zeros((n_models, n_models))
        
        pair_idx = 0
        
        for i in range(n_models):
            for j in range(i + 1, n_models):
                pair_idx += 1
                distance = self.compute_mw2_distance(model_paths[i], model_paths[j])
                distances.append(distance)
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance 
        
        distances = np.array(distances)
        
        sum_distances = np.sum(distances)
        sum_distances_squared = np.sum(distances ** 2)
        imr = np.log(sum_distances_squared / sum_distances)
        
        stability_metrics = {
            'imr': float(imr),
            'n_models': n_models,
            'n_comparisons': len(distances),
            'max_gaussians_used': self.max_gaussians,
        }
        return stability_metrics


def evaluate_stability(model_paths: List[str], 
                      max_gaussians: int = 10000,
                      device: str = 'cuda') -> Dict[str, float]:
    metric = MW2StabilityMetric(
        max_gaussians=max_gaussians,
        device=device,
    )
    
    return metric.evaluate_training_stability(model_paths)


def evaluate_stability_from_folder(folder_path: str,
                                  max_gaussians: int = 10000,
                                  device: str = 'cuda',
                                  output_csv: Optional[str] = None) -> Dict[str, float]:
    
    ply_pattern = os.path.join(folder_path, "**/*.ply")
    model_paths = glob.glob(ply_pattern, recursive=True)
    
    model_paths = [path for path in model_paths if not os.path.basename(path).lower() == 'input.ply']
    
    model_paths.sort()
    
    metrics = evaluate_stability(
        model_paths,
        max_gaussians=max_gaussians,
        device=device,
    )
    
    dataset_name = os.path.basename(folder_path)
    save_results_to_csv(metrics, dataset_name, output_csv, device)
    
    return metrics


def save_results_to_csv(metrics: Dict[str, float], 
                       dataset_name: str = "custom_models",
                       output_csv: Optional[str] = None,
                       device: str = 'cuda') -> None:
    
    csv_data = {
        'dataset_name': dataset_name,
        'n_models': metrics['n_models'],
        'n_comparisons': metrics['n_comparisons'],
        'max_gaussians_used': metrics['max_gaussians_used'],
        'IMR': metrics['imr'],
    }
    
    csv_path = f"mw2_stability_results_{dataset_name}.csv"
    
    file_exists = os.path.exists(csv_path)
    
    with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
        fieldnames = list(csv_data.keys())
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(csv_data)
    
    if output_csv:
        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = list(csv_data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            
            writer.writerow(csv_data)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--models', nargs='+')
    group.add_argument('--folder', type=str)
    parser.add_argument('--max-gaussians', type=int, default=10000)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-csv', type=str, default=None)
    args = parser.parse_args()
    
    if args.folder:
        folder_path = args.folder
        ply_pattern = os.path.join(folder_path, "**/*.ply")
        model_paths = glob.glob(ply_pattern, recursive=True)
        model_paths = [path for path in model_paths if not os.path.basename(path).lower() == 'input.ply']
        model_paths.sort()
        
    else:
        model_paths = args.models
    
    if args.folder:
        metrics = evaluate_stability_from_folder(
            args.folder,
            max_gaussians=args.max_gaussians,
            device=args.device,
            output_csv=args.output_csv,
        )
    else:
        metrics = evaluate_stability(
            model_paths, 
            max_gaussians=args.max_gaussians,
            device=args.device,
        )
        dataset_name = 'custom_models'
        save_results_to_csv(metrics, dataset_name, args.output_csv, args.device)
    
    print(f"IMR: {metrics['imr']:.6f}")
    return 0

if __name__ == "__main__":
    exit(main())

    