import os
import math
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement

import torch
from torch import nn

from .sh_utils import eval_sh, SH2RGB, RGB2SH
# from .mesh import Mesh
# from .mesh_utils import decimate_mesh, clean_mesh


def build_rotation(r):
    # TODO this is not batched implementation, need to be modified
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y*y + z*z)
    R[:, 0, 1] = 2 * (x*y - r*z)
    R[:, 0, 2] = 2 * (x*z + r*y)
    R[:, 1, 0] = 2 * (x*y + r*z)
    R[:, 1, 1] = 1 - 2 * (x*x + z*z)
    R[:, 1, 2] = 2 * (y*z - r*x)
    R[:, 2, 0] = 2 * (x*z - r*y)
    R[:, 2, 1] = 2 * (y*z + r*x)
    R[:, 2, 2] = 1 - 2 * (x*x + y*y)
    return R


class densification_module(torch.nn.Module):
    
    def __init__(self, num_points):
        super().__init__()
        
        # TODO save desification relevant tensors?
        self.xyz_gradient_accum = torch.zeros((num_points, 1))
        self.denom = torch.zeros((num_points, 1))
        self.max_radii2D = torch.zeros((num_points))
        self.opacity_accum = torch.zeros((num_points, 1))
    
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log
        self.opacity_activation = torch.sigmoid
        self.rotation_activation = torch.nn.functional.normalize
        
        self.percent_dense = 0.1
        self.clone_K = 1
        
        # TODO below tensors are overwritten in every update, since this way is easy to implement
        self.param_dict = {}
        self._xyz = None
        self._features_dc = None
        self._features_rest = None
        self._opacity = None
        self._scaling = None
        self._rotation = None
        
        self.latest_clone_mask = None
        self.latest_split_mask = None
    
    @property
    def get_scaling(self):
        return self.scaling_activation(self._scaling)
    
    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)
    
    @property
    def get_xyz(self):
        return self._xyz
    
    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)
    
    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)
    
    
    def _prune_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "name" not in group.keys(): # model (generator) parameters 
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def prune_points(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_optimizer(valid_points_mask)

        self._xyz = optimizable_tensors["xyz"]
        # self._scaling = optimizable_tensors["scaling"]
        # self._rotation = optimizable_tensors["rotation"]
        
        # self._features_dc = optimizable_tensors["f_dc"]
        # self._features_rest = optimizable_tensors["f_rest"]
        # self._opacity = optimizable_tensors["opacity"]

        self.xyz_gradient_accum = self.xyz_gradient_accum[valid_points_mask]
        self.denom = self.denom[valid_points_mask]
        self.max_radii2D = self.max_radii2D[valid_points_mask]
        self.opacity_accum = self.opacity_accum[valid_points_mask]

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if "name" not in group.keys(): # model (generator) parameters 
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            
            # assert stored_state == None, "store_state is not None" # TODO check
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def densification_postfix(self, new_xyz, new_features_dc=None, new_features_rest=None, new_opacities=None, new_scaling=None, new_rotation=None):
        d = {"xyz": new_xyz,}

        optimizable_tensors = self.cat_tensors_to_optimizer(d)
        self._xyz = optimizable_tensors["xyz"]

        self.xyz_gradient_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self._xyz.device)
        self.denom = torch.zeros((self.get_xyz.shape[0], 1), device=self._xyz.device)
        self.max_radii2D = torch.zeros((self.get_xyz.shape[0]), device=self._xyz.device)
        self.opacity_accum = torch.zeros((self.get_xyz.shape[0], 1), device=self._xyz.device)
        
    # # https://github.com/VITA-Group/FSGS/blob/main/scene/gaussian_model.py
    # def proximity(self, scene_extent, N = 3):
    #     dist, nearest_indices = distCUDA2(self.get_xyz)
    #     selected_pts_mask = torch.logical_and(dist > (5. * scene_extent),
    #                                           torch.max(self.get_scaling, dim=1).values > (scene_extent))

    #     new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
    #     source_xyz = self._xyz[selected_pts_mask].repeat(1, N, 1).reshape(-1, 3)
    #     target_xyz = self._xyz[new_indices]
    #     new_xyz = (source_xyz + target_xyz) / 2
    #     new_scaling = self._scaling[new_indices]
    #     new_rotation = torch.zeros_like(self._rotation[new_indices])
    #     new_rotation[:, 0] = 1
    #     new_features_dc = torch.zeros_like(self._features_dc[new_indices])
    #     new_features_rest = torch.zeros_like(self._features_rest[new_indices])
    #     new_opacity = self._opacity[new_indices]
    #     self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)

    def densify_and_split(self, grads, grad_threshold, scene_extent, N=2):
        n_init_points = self.get_xyz.shape[0]
        # Extract points that satisfy the gradient condition
        padded_grad = torch.zeros((n_init_points), device=self._xyz.device)
        padded_grad[:grads.shape[0]] = grads.squeeze()
        selected_pts_mask = torch.where(padded_grad >= grad_threshold, True, False)
        selected_pts_mask = torch.logical_and(selected_pts_mask,
                                              torch.max(self.get_scaling, dim=1).values > self.percent_dense*scene_extent)

        if str(selected_pts_mask.device) == "cuda:0":
            print("split:", selected_pts_mask.sum(), selected_pts_mask.shape)

        stds = self.get_scaling[selected_pts_mask].repeat(N,1)
        means = torch.zeros((stds.size(0), 3),device=self._xyz.device)
        samples = torch.normal(mean=means, std=stds)
        rots = build_rotation(self._rotation[selected_pts_mask]).repeat(N,1,1)
        new_xyz = torch.bmm(rots, samples.unsqueeze(-1)).squeeze(-1) + self.get_xyz[selected_pts_mask].repeat(N, 1)
        new_scaling = self.scaling_inverse_activation(self.get_scaling[selected_pts_mask].repeat(N,1) / (0.8*N))
        new_rotation = self._rotation[selected_pts_mask].repeat(N,1)
        
        
        new_features_dc = self._features_dc[selected_pts_mask].repeat(N,1,1) if self._features_dc is not None else None
        new_features_rest = self._features_rest[selected_pts_mask].repeat(N,1,1) if self._features_rest is not None else None
        new_opacity = self._opacity[selected_pts_mask].repeat(N,1) if self._opacity is not None else None

        # TODO for cloning ema param
        self.latset_split_mask = selected_pts_mask

        self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacity, new_scaling, new_rotation)
        
        prune_filter = torch.cat((selected_pts_mask, torch.zeros(N * selected_pts_mask.sum(), device=self._xyz.device, dtype=bool)))
        
        # TODO opacity_accum also needs to opacity accumulation
        self.opacity_accum_cur = torch.cat([self.opacity_accum_cur[torch.logical_not(selected_pts_mask)], \
                                    self.opacity_accum_cur[selected_pts_mask].repeat(N,1)], dim=0)
        # if str(selected_pts_mask.device) == "cuda:0":
        #     print("prune:", prune_filter.sum(), prune_filter.shape)
        self.prune_points(prune_filter)

    def densify_and_clone(self, grads, grad_threshold, scene_extent):
        
        # TODO unpool
        K = self.clone_K
        dist = (self._xyz.unsqueeze(0) - self._xyz.unsqueeze(1)).square().sum(-1).sqrt()
        dists, indices = torch.sort(dist, dim=-1)
        nearest_dists, nearest_indices = dists[:, 1:K+1], indices[:, 1:K+1]
        
        
        # Extract points that satisfy the gradient condition
        selected_pts_mask = torch.where(torch.norm(grads, dim=-1) >= grad_threshold, True, False)
        # selected_pts_mask = torch.logical_and(selected_pts_mask,
        #                                       nearest_dists.mean(dim=-1) >= 0.4) # TODO hard-coded threshold
        
        if str(selected_pts_mask.device) == "cuda:0":
            print("clone:", selected_pts_mask.sum(), selected_pts_mask.shape)
        
        param_dict = {}
        
        # new_indices = nearest_indices[selected_pts_mask].reshape(-1).long()
        # source_xyz = self._xyz[selected_pts_mask].repeat(1, K, 1).reshape(-1, 3)
        # target_xyz = self._xyz[new_indices]
        # new_xyz = (source_xyz + target_xyz) / 2
        
        new_xyz = self._xyz[selected_pts_mask] + torch.randn_like(self._xyz[selected_pts_mask]) * 0.03
        # new_scaling = self._scaling[selected_pts_mask]
        # new_rotation = self._rotation[selected_pts_mask]
        # new_features_dc = self._features_dc[selected_pts_mask] if self._features_dc is not None else None
        # new_features_rest = self._features_rest[selected_pts_mask] if self._features_rest is not None else None
        # new_opacities = self._opacity[selected_pts_mask] if self._opacity is not None else None

        # TODO opacity_accum also needs to opacity accumulation
        self.opacity_accum_cur = torch.cat([self.opacity_accum_cur] + [self.opacity_accum_cur[selected_pts_mask]] * K, dim=0)

        # TODO for cloning ema param
        self.latest_clone_mask = selected_pts_mask # it should be "added" to the original params

        self.densification_postfix(new_xyz)
        # self.densification_postfix(new_xyz, new_features_dc, new_features_rest, new_opacities, new_scaling, new_rotation)


    # ========================= below functions are called in training loop =========================
    
    # TODO check extent and max_screen_size
    # TODO check max_grad and min_opacity
    def densify_and_prune(self, gs_params, max_grad=2e-3, min_opacity=0.001, extent=1., max_screen_size=None):
    # def densify_and_prune(self, gs_params, max_grad=1e-2, min_opacity=0.001, extent=1., max_screen_size=None):
    # def densify_and_prune(self, gs_params, max_grad=5e-4, min_opacity=0.001, extent=1., max_screen_size=None):
    # def densify_and_prune(self, gs_params, max_grad=2e-4, min_opacity=0.001, extent=1., max_screen_size=None):
    # def densify_and_prune(self, gs_params, max_grad=2e-5, min_opacity=0.01, extent=1., max_screen_size=None):
        # TODO set params
        self.param_dict = gs_params
        self._xyz = gs_params["_xyz"]
        self._features_dc = None # not learnable
        self._features_rest = None # not learnable 
        self._opacity = None # not learnable
        # self._scaling = gs_params["_scale"]
        # self._rotation = gs_params["_rotation"]
        
        # This is main function for densification
        grads = self.xyz_gradient_accum / self.denom
        grads[grads.isnan()] = 0.0
        self.opacity_accum_cur = self.opacity_accum # it re-intialized in densify_and_*
        
        self.latest_clone_mask = None
        # self.latest_split_mask = None
        
        self.densify_and_clone(grads, max_grad, extent)
        # self.densify_and_split(grads, max_grad, extent)

        # # TODO currently, opacity cannot be calculated simply, cause they are different as the given latent codes
        # # TODO need to implement ema too
        # # prune_mask = (self.get_opacity < min_opacity).squeeze()
        
        # prune_mask = (self.opacity_accum_cur < min_opacity).squeeze()
        
        # # if max_screen_size:
        # #     big_points_vs = self.max_radii2D > max_screen_size
        # #     big_points_ws = self.get_scaling.max(dim=1).values > 0.1 * extent
        # #     prune_mask = torch.logical_or(torch.logical_or(prune_mask, big_points_vs), big_points_ws)
        
        # if str(prune_mask.device) == "cuda:0":
        #     print("prune:", prune_mask.sum(), prune_mask.shape)
        
        # self.prune_points(prune_mask)

        torch.cuda.empty_cache()
        return self._xyz, self._scaling, self._rotation
        

    def add_densification_stats(self, viewspace_point_tensor, update_filter, opacity, densify_inteval, cur_pts):
        self.xyz_gradient_accum = self.xyz_gradient_accum.to(opacity.device)
        self.denom = self.denom.to(opacity.device)
        self.opacity_accum = self.opacity_accum.to(opacity.device)
        
        
        viewpoint_grads = 0
        upfilter_agg = 0
        if type(viewspace_point_tensor) == list:
            # Average gradient along with the batch dimension
            for vpt, upfilter in zip(viewspace_point_tensor, update_filter):
                viewpoint_grads += torch.norm(vpt.grad[:,:2] * upfilter.unsqueeze(-1), dim=-1, keepdim=True) / len(viewspace_point_tensor)
                upfilter_agg += upfilter.unsqueeze(-1)
                
        nblocks = 5
        # prev_pts, cur_pts = 0, opacity.shape[1]
        prev_pts, cur_pts = 0, cur_pts
        up_ratio = 4
        xyz_grads = 0
        
        # update_filter = update_filter.unsqueeze(-1)
        # viewpoint_grads = viewpoint_grads * update_filter
        opacity_avg = self.opacity_activation(opacity).mean(0)
        
        for i in range(int(nblocks)):
            xyz_grads = xyz_grads + viewpoint_grads[prev_pts:prev_pts + cur_pts].reshape([cur_pts // up_ratio**i, up_ratio**i, -1]).mean(1) / nblocks
            self.denom += upfilter_agg[prev_pts:prev_pts + cur_pts].reshape([cur_pts // up_ratio**i, up_ratio**i, -1]).float().mean(1) / nblocks
            
            self.opacity_accum += opacity_avg[prev_pts:prev_pts + cur_pts].reshape([cur_pts // up_ratio**i, up_ratio**i, -1]).float().mean(1) / densify_inteval / nblocks
            cur_pts = cur_pts * up_ratio
            
            # # break at first block (256)
            # break
        
        # if str(opacity.device) == 'cuda:0':
        #     print(xyz_grads.mean(), xyz_grads.std(), xyz_grads.max(), xyz_grads.min())
        self.xyz_gradient_accum += xyz_grads
        # self.denom += 1
        
        # TODO as denom exist, we do not divided xyz_grads by densify_interval
        
        # # TODO opacity is added. weight sum? or just sum?
        # self.opacity_accum += self.opacity_activation(opacity).mean(0) / densify_inteval
        
        return xyz_grads.mean(), xyz_grads.std(), xyz_grads.max(), xyz_grads.min()
        

    def add_densification_stats_maxgrad(self, viewspace_point_tensor, update_filter, opacity, densify_inteval, cur_pts):
        self.xyz_gradient_accum = self.xyz_gradient_accum.to(opacity.device)
        self.denom = self.denom.to(opacity.device)
        self.opacity_accum = self.opacity_accum.to(opacity.device)
        
        
        viewpoint_grads = 0
        upfilter_agg = 0
        if type(viewspace_point_tensor) == list:
            # Maximum gradient along with the batch dimension
            # for vpt, upfilter in zip(viewspace_point_tensor, update_filter):
            #     viewpoint_grads = torch.norm(vpt.grad[:,:2] * upfilter.unsqueeze(-1), dim=-1, keepdim=True)
            #     upfilter_agg += upfilter.unsqueeze(-1)
            upfilter = torch.stack(update_filter).unsqueeze(-1)
            upfilter_agg = torch.sum(upfilter, dim=0)
            viewpoint_grads = torch.norm(torch.stack(viewspace_point_tensor).grad[:, :,:2] * upfilter, dim=-1, keepdim=True).max(0)
                
        nblocks = 5
        # prev_pts, cur_pts = 0, opacity.shape[1]
        prev_pts, cur_pts = 0, cur_pts
        up_ratio = 4
        xyz_grads = 0
        
        # update_filter = update_filter.unsqueeze(-1)
        # viewpoint_grads = viewpoint_grads * update_filter
        opacity_avg = self.opacity_activation(opacity).mean(0)
        
        for i in range(int(nblocks)):
            xyz_grads = xyz_grads + viewpoint_grads[prev_pts:prev_pts + cur_pts].reshape([cur_pts // up_ratio**i, up_ratio**i, -1]).mean(1) / nblocks
            self.denom += upfilter_agg[prev_pts:prev_pts + cur_pts].reshape([cur_pts // up_ratio**i, up_ratio**i, -1]).float().mean(1) / nblocks
            
            self.opacity_accum += opacity_avg[prev_pts:prev_pts + cur_pts].reshape([cur_pts // up_ratio**i, up_ratio**i, -1]).float().mean(1) / densify_inteval / nblocks
            cur_pts = cur_pts * up_ratio
            
            # # break at first block (256)
            # break
        
        # if str(opacity.device) == 'cuda:0':
        #     print(xyz_grads.mean(), xyz_grads.std(), xyz_grads.max(), xyz_grads.min())
        # self.xyz_gradient_accum += xyz_grads
        self.xyz_gradient_accum = torch.max(self.xyz_gradient_accum, xyz_grads)
        self.denom = 1
        # self.denom += 1
        
        # TODO as denom exist, we do not divided xyz_grads by densify_interval
        
        # # TODO opacity is added. weight sum? or just sum?
        # self.opacity_accum += self.opacity_activation(opacity).mean(0) / densify_inteval
        
        return xyz_grads.mean(), xyz_grads.std(), xyz_grads.max(), xyz_grads.min()
        
    
    def add_densification_stats_original(self, viewspace_point_tensor, update_filter, opacity, densify_inteval, cur_pts):
        self.xyz_gradient_accum = self.xyz_gradient_accum.to(update_filter.device)
        self.denom = self.denom.to(update_filter.device)
        self.max_radii2D = self.max_radii2D.to(update_filter.device)
        self.opacity_accum = self.opacity_accum.to(update_filter.device)
        
        if type(viewspace_point_tensor) == list:
            for vpt in viewspace_point_tensor:
                self.xyz_gradient_accum[update_filter] += torch.norm(vpt.grad[update_filter,:2], dim=-1, keepdim=True) / len(viewspace_point_tensor)
        else:
            self.xyz_gradient_accum[update_filter] += torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.denom[update_filter] += 1
        
        # TODO opacity is added. weight sum? or just sum?
        self.opacity_accum += self.opacity_activation(opacity).mean(0) / densify_inteval
        
    
    def set_optimizer(self, optimizer):
        self.optimizer = optimizer