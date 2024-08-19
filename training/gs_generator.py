save_ply# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import torch
from torch_utils import persistence
from training.networks_stylegan2 import Generator as StyleGAN2Backbone, MappingNetwork
import dnnlib

from training.gaussian3d_splatting.renderer import Renderer, MiniCam, BasicPointCloud, inverse_sigmoid
from training.gaussian3d_splatting.gaussian_model import densification_module
from training.gaussian3d_splatting.sh_utils import eval_sh, SH2RGB, RGB2SH
from training.gaussian3d_splatting.cam_utils import orbit_camera, OrbitCamera

# # for debugging
# import cv2
import numpy as np
from training.point_generator import LFF, ModulatedFullyConnectedLayer, PointGenerator
from training.background_network import PointBgGenerator
from custom_utils import save_ply
import numpy as np

@persistence.persistent_class
class GSGenerator(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality.
        c_dim,                      # Conditioning label (C) dimensionality.
        w_dim,                      # Intermediate latent (W) dimensionality.
        img_resolution,             # Output resolution.
        img_channels,               # Number of output color channels.
        sr_num_fp16_res     = 0,
        mapping_kwargs      = {},   # Arguments for MappingNetwork.
        rendering_kwargs    = {},
        sr_kwargs = {},
        **synthesis_kwargs,         # Arguments for SynthesisNetwork.
    ):
        super().__init__()
        self.z_dim=z_dim
        self.c_dim=c_dim
        self.w_dim=w_dim
        self.img_resolution=img_resolution
        self.img_channels=img_channels
        
        self.neural_rendering_resolution = img_resolution
        self.rendering_kwargs = rendering_kwargs
        self.custom_options = rendering_kwargs['custom_options']
    
        self._last_planes = None
    
        # ========== init gaussian splatting ==========
        """
        Initialize points uniformly in box
        """
        
        init_pts = self.custom_options['num_pts']
        self.num_pts = init_pts
        
        init_dim = 3 # xyz 
        random_coords = torch.randn((init_pts, init_dim))
        self._xyz = torch.nn.Parameter(random_coords * torch.rsqrt(torch.mean(random_coords ** 2, dim=1, keepdim=True) + 1e-8) * 0.5)
        
        self.point_gen = PointGenerator(w_dim=w_dim, img_channels=3, img_resolution=img_resolution, options=self.custom_options, \
                                            init_pts=init_pts, init_dim=init_dim, use_dir_cond=self.custom_options['use_dir_cond'])
        
        random_coords = torch.randn((2000, 3))
        
        # apply absolute in z dimension
        random_coords = torch.cat([random_coords[:, :2], -torch.abs(random_coords[:, 2:3])], dim=1)
        
        self._xyz_bg = torch.nn.Parameter(random_coords * torch.rsqrt(torch.mean(random_coords ** 2, dim=1, keepdim=True) + 1e-8))
        self.point_gen_bg = PointBgGenerator(w_dim=w_dim, img_channels=3, options=rendering_kwargs['custom_options'], num_upsample=1)
        
        # 3d gaussian splatting
        self.renderer_gaussian3d = Renderer(sh_degree=0) # do not use spherical harmonics
        
        # Emprically found camera rotation (EG3D -> 3D gaussian splatting settings)
        self.reverse_camera_direction = torch.tensor(
                        [[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]],
                        dtype=torch.float32,
        )
        
        self.camera_noise_std = rendering_kwargs['custom_options']['camera_noise_std']
        self._last_gaussians = None
        
        self.camera_cond_origin = torch.tensor([0, 0, 1], dtype=torch.float32)
        # =========================================
    
        self.mapping_network = MappingNetwork(z_dim=z_dim, c_dim=c_dim, w_dim=w_dim, num_ws=self.point_gen.num_ws + 1, **mapping_kwargs) # +1 for background
    
    def update_gaussian_params(self, _xyz, _scale, _rotation):
        self._xyz = _xyz
        self.num_pts = _xyz.shape[0]
        
    
    def mapping(self, z, c, truncation_psi=1, truncation_cutoff=None, update_emas=False):
        if self.rendering_kwargs['c_gen_conditioning_zero']:
            c = torch.zeros_like(c)
        # return self.backbone.mapping(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.mapping_network(z, c * self.rendering_kwargs.get('c_scale', 0), truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)


    # This is code snippet for checking the minimal implementation of the generator
    def synthesis_minimal(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        ws_fg, ws_bg = ws[:, :-1], ws[:, -1]
        
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution
        

        # ================= Rendering part ==================
        # change camera parameters (EG3D -> 3D gaussian splatting)
        pose = torch.bmm(cam2world_matrix, self.reverse_camera_direction.unsqueeze(0)\
                                        .repeat(len(cam2world_matrix), 1, 1).to(cam2world_matrix.device))
        
        focalx, focaly, near, far = intrinsics[:, 0,0], intrinsics[:, 1,1], self.rendering_kwargs['ray_start'], self.rendering_kwargs['ray_end']
        
        # TODO EG3D assumes focal length 4.26 x image width, this makes fov ~ 12.96
        fovx, fovy, near, far = 12.96, 12.96, near, far

        features_imgs = []
        depth_imgs = []
        alpha_imgs = []

        
        if use_cached_backbone and self._last_gaussians is not None:
            dec_out = self._last_gaussians
        else:
            sample_coordinates = torch.tanh(self._xyz.unsqueeze(0).repeat(len(ws), 1, 1))
            
            camera_cond = torch.nn.functional.normalize(camera_cond, dim=-1)
            sample_coordinates, sample_scale, sample_rotation, sample_color, sample_opacity, anchors = self.point_gen(sample_coordinates, ws_fg, \
                                                    camera_cond=camera_cond)
            point_gen_coords = sample_coordinates
            
            # ============================================================================
            
            dec_out = {}
            dec_out["sample_coordinates"] = sample_coordinates
            dec_out["scale"] = sample_scale
            dec_out["rotation"] = sample_rotation
            dec_out["color"] = sample_color
            dec_out["opacity"] = sample_opacity
            
                
        if cache_backbone:
            self._last_gaussians = dec_out

        for batch_idx in range(len(ws)):
            c_idx = 0
            gaussian_params = {}

            gaussian_params["_xyz"] = dec_out['sample_coordinates'][batch_idx]
            gaussian_params["_features_dc"] = dec_out["color"][batch_idx].unsqueeze(1).contiguous() # self._features_dc # 3
            gaussian_params["_features_rest"] = dec_out["color"][batch_idx].unsqueeze(1)[:, 0:0].contiguous() # self._features_rest # 3
            gaussian_params["_scaling"] = dec_out["scale"][batch_idx] # self._scaling # 3
            gaussian_params["_rotation"] = dec_out["rotation"][batch_idx] # self._rotation # 4
            gaussian_params["_opacity"] = dec_out["opacity"][batch_idx] # self._opacity # 1
            
            cur_cam = MiniCam(
                pose[batch_idx],
                neural_rendering_resolution,
                neural_rendering_resolution,
                fovy,
                fovx,
                near,
                far,
                device=ws.device
            )
            
            invert_bg_color = False
            ret_dict = self.renderer_gaussian3d.render(gaussian_params, cur_cam, invert_bg_color=invert_bg_color)
            
            features_imgs.append(ret_dict["image"].unsqueeze(0))
            depth_imgs.append(ret_dict["depth"].unsqueeze(0))
            alpha_imgs.append(ret_dict["alpha"].unsqueeze(0))
            
        feature_image = torch.cat(features_imgs, dim=0).to(ws.device)
        depth_image = torch.cat(depth_imgs, dim=0).to(ws.device)
        alpha_image = torch.cat(alpha_imgs, dim=0).to(ws.device)
        
        rgb_image = feature_image[:, :3]
            
        # ===================================================
        sr_image = rgb_image
        
        
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, \
                "scale": dec_out["scale"], "opacity": dec_out["opacity"], "scale": sample_scale, "sample_coordinates": point_gen_coords, 'image_alpha': alpha_image, \
                'init_anchors': anchors[0][:, :self.num_pts], 'anchors': anchors[0], 'gaussian_params': gaussian_params, 'dec_out': dec_out}


    def synthesis(self, ws, c, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, gs_params=None, **synthesis_kwargs):
        ws_fg, ws_bg = ws[:, :-1], ws[:, -1]
        
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25].view(-1, 3, 3)

        if neural_rendering_resolution is None:
            neural_rendering_resolution = self.neural_rendering_resolution
        else:
            self.neural_rendering_resolution = neural_rendering_resolution
        

        # ================= Rendering part ==================
        # change camera parameters (EG3D -> 3D gaussian splatting)
        pose = torch.bmm(cam2world_matrix, self.reverse_camera_direction.unsqueeze(0)\
                                        .repeat(len(cam2world_matrix), 1, 1).to(cam2world_matrix.device))
        
        focalx, focaly, near, far = intrinsics[:, 0,0], intrinsics[:, 1,1], self.rendering_kwargs['ray_start'], self.rendering_kwargs['ray_end']
        
        # TODO EG3D assumes focal length 4.26 x image width, this makes fov ~ 12.96
        fovx, fovy, near, far = 12.96, 12.96, near, far

        features_imgs = []
        depth_imgs = []
        alpha_imgs = []
        
        # Parameters for visualization
        gs_params_default = {
            "rendering_scale": 0, # should be zero for training
            "res_visualize": None, # [0, 1, 2, ...], visualizing gaussians in those block indices
            "disable_background": False,
            'opacity_ones': False,
            'point_index': [],
            'num_init_near_point': -1,
            'save_ply': None,
            'visualize_anchor': False,
            'camera_cond': None,
            'cam_interp_ratio': 0., # always not use center-oriented camera [0, 1]
            'cam_dir_swap_p': 0.,
        }
        if gs_params is not None:
            for k in gs_params.keys():
                gs_params_default[k] = gs_params[k]
        gs_params = gs_params_default
        
        if use_cached_backbone and self._last_gaussians is not None:
            dec_out = self._last_gaussians
        else:
            sample_coordinates = torch.tanh(self._xyz.unsqueeze(0).repeat(len(ws), 1, 1))
            
            if gs_params['num_init_near_point'] > 0:
                point_index = gs_params['point_index']
                assert len(point_index) == 1, f"{gs_params['point_index']}, when visualize near points, you should only give a singe point index as query"
                distance = (self._xyz[point_index[0]:point_index[0]+1] - self._xyz).square().sum(-1)
                near_idx = torch.argsort(distance)[:gs_params['num_init_near_point']]
                gs_params['point_index'] = near_idx # only for visualization
                gs_params['num_init_near_point'] = -1
                
            if gs_params['camera_cond'] is None:
                camera_cond = -pose[:, :3,  2] # TODO direction toward (0, 0, 0) of world cooridante
            else:
                camera_cond = torch.tensor(gs_params['camera_cond']).to(device=ws.device, dtype=ws.dtype)
            cam_interp_ratio = gs_params['cam_interp_ratio']
            if cam_interp_ratio > 0:
                camera_cond_origin = -self.camera_cond_origin.to(device=camera_cond.device, dtype=camera_cond.dtype).unsqueeze(0).repeat(camera_cond.shape[0], 1)
                camera_cond = slerp(camera_cond, camera_cond_origin, cam_interp_ratio)
            
            camera_cond = torch.nn.functional.normalize(camera_cond, dim=-1)
                
            sample_coordinates, sample_scale, sample_rotation, sample_color, sample_opacity, anchors = self.point_gen(sample_coordinates, ws_fg, \
                                                    camera_cond=camera_cond, camera_swap_p=gs_params['cam_dir_swap_p'])
            
            
            # =========================== TODO Post processing for visualization ===========================
            # TODO visualize anchor
            if gs_params['visualize_anchor']:
                sample_coordinates, sample_scale, sample_rotation, sample_color, sample_opacity = anchors
                gs_params['opacity_ones'] = True
            
            point_gen_coords = sample_coordinates
            sample_scale = sample_scale + gs_params['rendering_scale']
            
            if gs_params["res_visualize"] is not None:
                prev_pts, cur_pts = 0, self.num_pts
                nblocks = np.log2(self.custom_options["res_end"] / 4) - np.log2(self.custom_options["res_upsample"] / 4)
                coords_per_res, scale_per_res, rotation_per_res, color_per_res, opacity_per_res = [], [], [], [], []
                
                ups = 1
                out_mult = self.custom_options['output_multiplier']
                for i in range(int(nblocks)):
                    if i in gs_params['res_visualize']:
                        if len(gs_params['point_index']) == 0:
                            coords_per_res.append(sample_coordinates[:, prev_pts:prev_pts + cur_pts])
                            scale_per_res.append(sample_scale[:, prev_pts:prev_pts + cur_pts])
                            rotation_per_res.append(sample_rotation[:, prev_pts:prev_pts + cur_pts])
                            color_per_res.append(sample_color[:, prev_pts:prev_pts + cur_pts])
                            opacity_per_res.append(sample_opacity[:, prev_pts:prev_pts + cur_pts])
                        else:
                            for idx_pt in gs_params['point_index']:
                                coords_per_res.append(sample_coordinates[:, out_mult * (prev_pts + ups * idx_pt):out_mult * (prev_pts+ups * (idx_pt + 1))])
                                scale_per_res.append(sample_scale[:, out_mult * (prev_pts + ups * idx_pt):out_mult * (prev_pts+ups * (idx_pt + 1))])
                                rotation_per_res.append(sample_rotation[:, out_mult * (prev_pts + ups * idx_pt):out_mult * (prev_pts+ups * (idx_pt + 1))])
                                color_per_res.append(sample_color[:, out_mult * (prev_pts + ups * idx_pt):out_mult * (prev_pts+ups * (idx_pt + 1))])
                                opacity_per_res.append(sample_opacity[:, out_mult * (prev_pts + ups * idx_pt):out_mult * (prev_pts+ups * (idx_pt + 1))])
                    
                    prev_pts = prev_pts + cur_pts
                    cur_pts = cur_pts * self.custom_options['up_ratio']
                    ups *= self.custom_options['up_ratio']
                    
                sample_coordinates = torch.cat(coords_per_res, dim=1)
                sample_scale = torch.cat(scale_per_res, dim=1)
                sample_rotation = torch.cat(rotation_per_res, dim=1)
                sample_color = torch.cat(color_per_res, dim=1)
                sample_opacity = torch.cat(opacity_per_res, dim=1)
                
            if gs_params["opacity_ones"]:
                sample_opacity = torch.ones_like(sample_opacity)
            # ============================================================================
            
            
            # background generator
            if not gs_params["disable_background"]:
                sample_coords_bg, sample_scale_bg, sample_rotation_bg, sample_color_bg, sample_opacity_bg = self.point_gen_bg(self._xyz_bg.unsqueeze(0).repeat(len(ws), 1, 1), ws_bg)
                
                sample_coordinates = torch.cat([sample_coordinates, sample_coords_bg], dim=1)
                sample_scale = torch.cat([sample_scale, sample_scale_bg], dim=1)
                sample_rotation = torch.cat([sample_rotation, sample_rotation_bg], dim=1)
                sample_color = torch.cat([sample_color, sample_color_bg], dim=1)
                sample_opacity = torch.cat([sample_opacity, sample_opacity_bg], dim=1)
            
            
            dec_out = {}
            dec_out["sample_coordinates"] = sample_coordinates
            dec_out["scale"] = sample_scale
            dec_out["rotation"] = sample_rotation
            dec_out["color"] = sample_color
            dec_out["opacity"] = sample_opacity
            
                
        if cache_backbone:
            self._last_gaussians = dec_out


        collected_stat = {
            "viewspace_points": [],
            "visibility_filter": [],
            "radii": [],
        }
        for batch_idx in range(len(ws)):
            c_idx = 0
            gaussian_params = {}

            gaussian_params["_xyz"] = dec_out['sample_coordinates'][batch_idx]
            gaussian_params["_features_dc"] = dec_out["color"][batch_idx].unsqueeze(1).contiguous() # self._features_dc # 3
            gaussian_params["_features_rest"] = dec_out["color"][batch_idx].unsqueeze(1)[:, 0:0].contiguous() # self._features_rest # 3
            gaussian_params["_scaling"] = dec_out["scale"][batch_idx] # self._scaling # 3
            gaussian_params["_rotation"] = dec_out["rotation"][batch_idx] # self._rotation # 4
            gaussian_params["_opacity"] = dec_out["opacity"][batch_idx] # self._opacity # 1
            
            cur_cam = MiniCam(
                pose[batch_idx],
                neural_rendering_resolution,
                neural_rendering_resolution,
                fovy,
                fovx,
                near,
                far,
                device=ws.device
            )
            
            invert_bg_color = False
            ret_dict = self.renderer_gaussian3d.render(gaussian_params, cur_cam, invert_bg_color=invert_bg_color)

            collected_stat["viewspace_points"].append(ret_dict["viewspace_points"])
            collected_stat["visibility_filter"].append(ret_dict["visibility_filter"])
            collected_stat["radii"].append(ret_dict["radii"])
            
            features_imgs.append(ret_dict["image"].unsqueeze(0))
            depth_imgs.append(ret_dict["depth"].unsqueeze(0))
            alpha_imgs.append(ret_dict["alpha"].unsqueeze(0))
            
        feature_image = torch.cat(features_imgs, dim=0).to(ws.device)
        depth_image = torch.cat(depth_imgs, dim=0).to(ws.device)
        alpha_image = torch.cat(alpha_imgs, dim=0).to(ws.device)
        
        for k in collected_stat.keys():
            if k == "visibility_filter":
                pass
            elif k == "radii":
                collected_stat[k] = torch.mean(torch.stack(collected_stat[k], dim=0).float(), dim=0).int()
            else:
                pass
        collected_stat["opacity"] = dec_out["opacity"].detach()
        
        rgb_image = feature_image[:, :3]
        
        # Save intermediate results for debugging
        # if len(ws) > 1:
        #     save_images(rgb_image, depth_image, device=rgb_image.device)
            
        # ===================================================
        sr_image = rgb_image
        
        # export gaussian parameters
        if gs_params['save_ply'] is not None: 
            save_ply(gaussian_params["_xyz"], gaussian_params["_features_dc"], gaussian_params["_features_rest"], gaussian_params["_scaling"], \
                gaussian_params["_opacity"], gaussian_params["_rotation"], gs_params['save_ply'])
        
        return {'image': sr_image, 'image_raw': rgb_image, 'image_depth': depth_image, 'collected_stat': collected_stat, \
                "scale": dec_out["scale"], "opacity": dec_out["opacity"], "scale": sample_scale, "sample_coordinates": point_gen_coords, 'image_alpha': alpha_image, \
                'init_anchors': anchors[0][:, :self.num_pts], 'anchors': anchors[0], 'gaussian_params': gaussian_params, 'dec_out': dec_out}

    def forward(self, z, c, c_mapping=None, truncation_psi=1, truncation_cutoff=None, neural_rendering_resolution=None, update_emas=False, cache_backbone=False, use_cached_backbone=False, **synthesis_kwargs):
        # Render a batch of generated images.
        if c_mapping is None:
            c_mapping = c
        ws = self.mapping(z, c_mapping, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff, update_emas=update_emas)
        return self.synthesis(ws, c, update_emas=update_emas, neural_rendering_resolution=neural_rendering_resolution, cache_backbone=cache_backbone, use_cached_backbone=use_cached_backbone, **synthesis_kwargs)


def print_stats(_dict):
    for k in _dict.keys():
        if k == "_features_rest":
            continue
        print("{} shape: {}, min: {} max: {}".format(k, _dict[k].shape, _dict[k].min(), _dict[k].max()))


def save_images(rgb_image, depth_image, device=None):
    if str(device) == "cuda:0":
        # save intermediate image for debugging
        temp_img = rgb_image.detach().cpu().numpy()[0]
        temp_img = np.clip(((temp_img + 1)* 127.5), 0, 255)
        cv2.imwrite("temp_saves/temp_G_{}_{}.jpg".format(str(rgb_image.device), 0), temp_img.transpose([1, 2, 0])[:, :, ::-1])
        temp_img = rgb_image.detach().cpu().numpy()[1]
        temp_img = np.clip(((temp_img + 1)* 127.5), 0, 255)
        cv2.imwrite("temp_saves/temp_G_{}_{}.jpg".format(str(rgb_image.device), 1), temp_img.transpose([1, 2, 0])[:, :, ::-1])
        
        temp_depth = depth_image.detach().cpu().numpy()[0]
        temp_depth = (temp_depth - temp_depth.min()) / (temp_depth.max() - temp_depth.min()) * 255
        cv2.imwrite("temp_saves/depth_G_{}_{}.jpg".format(str(depth_image.device), 0), temp_depth.transpose([1, 2, 0])[:, :, ::-1])
    

def slerp(v0, v1, t):
    v0 = torch.nn.functional.normalize(v0, p=2, dim=-1)
    v1 = torch.nn.functional.normalize(v1, p=2, dim=-1)

    dot_product = torch.einsum('bi,bi->b', v0, v1)
    dot_product = torch.clamp(dot_product, -1.0, 1.0)
    theta = torch.acos(dot_product)
    
    sin_theta = torch.sin(theta)
    interpolated_vector = (torch.sin((1 - t) * theta) / sin_theta)[:, None] * v0 + (torch.sin(t * theta) / sin_theta)[:, None] * v1

    return interpolated_vector
