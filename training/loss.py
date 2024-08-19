# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Loss functions."""

import numpy as np
import torch
from torch_utils import training_stats
from torch_utils.ops import conv2d_gradfix
from torch_utils.ops import upfirdn2d
from training.dual_discriminator import filtered_resizing

# import wandb
from camera_utils import UniformCameraPoseSampler, GaussianCameraPoseSampler, CustomCameraPoseSampler
from training.gaussian3d_splatting.renderer import build_covariance_from_scaling_rotation_cov
import cv2

from torch.nn.utils.clip_grad import clip_grad_norm_

#----------------------------------------------------------------------------

class Loss:
    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg): # to be overridden by subclass
        raise NotImplementedError()

#----------------------------------------------------------------------------

class StyleGAN2Loss(Loss):
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0, r1_gamma_init=0, r1_gamma_fade_kimg=0, neural_rendering_resolution_initial=64, neural_rendering_resolution_final=None, neural_rendering_resolution_fade_kimg=0, gpc_reg_fade_kimg=1000, gpc_reg_prob=None, dual_discrimination=False, filter_mode='antialiased', loss_custom_options={}):
        super().__init__()
        self.device             = device
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma
        self.style_mixing_prob  = style_mixing_prob
        self.pl_weight          = pl_weight
        self.pl_batch_shrink    = pl_batch_shrink
        self.pl_decay           = pl_decay
        self.pl_no_weight_grad  = pl_no_weight_grad
        self.pl_mean            = torch.zeros([], device=device)
        self.blur_init_sigma    = blur_init_sigma
        self.blur_fade_kimg     = blur_fade_kimg
        self.r1_gamma_init      = r1_gamma_init
        self.r1_gamma_fade_kimg = r1_gamma_fade_kimg
        self.neural_rendering_resolution_initial = neural_rendering_resolution_initial
        self.neural_rendering_resolution_final = neural_rendering_resolution_final
        self.neural_rendering_resolution_fade_kimg = neural_rendering_resolution_fade_kimg
        self.gpc_reg_fade_kimg = gpc_reg_fade_kimg
        self.gpc_reg_prob = gpc_reg_prob
        self.dual_discrimination = dual_discrimination
        self.filter_mode = filter_mode
        self.resample_filter = upfirdn2d.setup_filter([1,3,3,1], device=device)
        self.blur_raw_target = True
        assert self.gpc_reg_prob is None or (0 <= self.gpc_reg_prob <= 1)
        
        
        self.coeffs = loss_custom_options
        
        # Camera rotation (EG3D -> 3D gaussian splatting settings)
        self.reverse_camera_direction = torch.tensor(
                        [[1, 0, 0, 0],
                        [0, -1, 0, 0],
                        [0, 0, -1, 0],
                        [0, 0, 0, 1]],
                        dtype=torch.float32,
        )
    
    def eg3d_to_3dgs(self, c):
        cam2world_matrix = c[:, :16].view(-1, 4, 4)
        intrinsics = c[:, 16:25]
        
        # ================= Rendering part ==================
        # change camera parameters (EG3D -> 3D gaussian splatting)
        pose = torch.bmm(cam2world_matrix, self.reverse_camera_direction.unsqueeze(0)\
                                        .repeat(len(cam2world_matrix), 1, 1).to(cam2world_matrix.device))
        c_new = torch.cat([pose.reshape([c.shape[0], -1]), intrinsics], dim=-1)
        
        return c_new
    
    def sample_custom_camera(self, c, p=0.2, horizontal_std=30/180, vertical_std=0.35, radius=2.7, htype='gaussian', vtype='uniform'):
        if p == 0.0:
            return c
        
        # As it is for FFHQ, use stat from FFHQ
        c_random = CustomCameraPoseSampler.sample(3.14/2, 3.14/2 - 5.159614799787766 / 180 * 3.14, horizontal_stddev=3.14*horizontal_std, vertical_stddev=3.14*vertical_std, radius=radius, batch_size=c.shape[0], device=c.device, \
                                                    horizontal_type=htype, vertical_type=vtype)
        
        c_random = c_random.reshape(c.shape[0], -1) # [B, 16] extrinsics only
        intrinsic = c[:, 16:].clone()
        c_random = torch.cat([c_random, intrinsic], dim=-1)
        
        selected_idx = (torch.rand(c.shape[0]) < p).unsqueeze(1).to(device=c.device, dtype=c.dtype)
        c_new = c * (1 - selected_idx) + c_random * selected_idx 
        
        return c_new

    def sample_uniform_camera(self, c, p=0.2, horizontal_std=0.5, vertical_std=0.35, radius=2.7):
        if p == 0.0:
            return c
        
        c_random = UniformCameraPoseSampler.sample(3.14/2, 3.14/2, horizontal_stddev=3.14*horizontal_std, vertical_stddev=3.14*vertical_std, radius=radius, batch_size=c.shape[0], device=c.device)
        
        c_random = c_random.reshape(c.shape[0], -1) # [B, 16] extrinsics only
        intrinsic = c[:, 16:].clone()
        c_random = torch.cat([c_random, intrinsic], dim=-1)
        
        selected_idx = (torch.rand(c.shape[0]) < p).unsqueeze(1).to(device=c.device, dtype=c.dtype)
        c_new = c * (1 - selected_idx) + c_random * selected_idx 
        
        return c_new
    
    def sample_gaussian_camera(self, c, p=0.2, horizontal_std=30/180, vertical_std=15/180, radius=2.7):
        if p == 0.0:
            return c
        
        c_random = GaussianCameraPoseSampler.sample(3.14/2, 3.14/2, horizontal_stddev=3.14*horizontal_std, vertical_stddev=3.14*vertical_std, radius=radius, batch_size=c.shape[0], device=c.device)
        
        c_random = c_random.reshape(c.shape[0], -1) # [B, 16] extrinsics only
        intrinsic = c[:, 16:].clone()
        c_random = torch.cat([c_random, intrinsic], dim=-1)
        
        selected_idx = (torch.rand(c.shape[0]) < p).unsqueeze(1).to(device=c.device, dtype=c.dtype)
        c_new = c * (1 - selected_idx) + c_random * selected_idx 
        
        return c_new
        
        
    def run_G(self, z, c, swapping_prob, neural_rendering_resolution, update_emas=False, gs_params=None):
        if swapping_prob is not None:
            c_swapped = torch.roll(c.clone(), 1, 0)
            c_gen_conditioning = torch.where(torch.rand((c.shape[0], 1), device=c.device) < swapping_prob, c_swapped, c)
        else:
            c_gen_conditioning = torch.zeros_like(c)

        ws = self.G.mapping(z, c_gen_conditioning, update_emas=update_emas)
        if self.style_mixing_prob > 0:
            with torch.autograd.profiler.record_function('style_mixing'):
                cutoff = torch.empty([], dtype=torch.int64, device=ws.device).random_(1, ws.shape[1])
                cutoff = torch.where(torch.rand([], device=ws.device) < self.style_mixing_prob, cutoff, torch.full_like(cutoff, ws.shape[1]))
                ws[:, cutoff:] = self.G.mapping(torch.randn_like(z), c, update_emas=False)[:, cutoff:]
        gen_output = self.G.synthesis(ws, c, neural_rendering_resolution=neural_rendering_resolution, update_emas=update_emas, gs_params=gs_params)
        return gen_output, ws

    def run_D(self, img, c, blur_sigma=0, blur_sigma_raw=0, update_emas=False, return_aux=False, force_augment=False):
        blur_size = np.floor(blur_sigma * 3)
        if blur_size > 0:
            with torch.autograd.profiler.record_function('blur'):
                f = torch.arange(-blur_size, blur_size + 1, device=img['image'].device).div(blur_sigma).square().neg().exp2()
                img['image'] = upfirdn2d.filter2d(img['image'], f / f.sum())

        if self.augment_pipe is not None and not force_augment:
            augmented_pair = self.augment_pipe(torch.cat([img['image'],
                                                    torch.nn.functional.interpolate(img['image_raw'], size=img['image'].shape[2:], mode='bilinear', antialias=True)],
                                                    dim=1))
            img['image'] = augmented_pair[:, :img['image'].shape[1]]
            img['image_raw'] = torch.nn.functional.interpolate(augmented_pair[:, img['image'].shape[1]:], size=img['image_raw'].shape[2:], mode='bilinear', antialias=True)

        logits, aux_out = self.D(img, c, update_emas=update_emas)
        
        if return_aux:
            return logits, aux_out
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gen_c, gain, cur_nimg, num_gpus):
        assert phase in ['Gmain', 'Greg', 'Gboth', 'Dmain', 'Dreg', 'Dboth']
        if self.G.rendering_kwargs.get('density_reg', 0) == 0:
            phase = {'Greg': 'none', 'Gboth': 'Gmain'}.get(phase, phase)
        if self.r1_gamma == 0:
            phase = {'Dreg': 'none', 'Dboth': 'Dmain'}.get(phase, phase)
        blur_sigma = max(1 - cur_nimg / (self.blur_fade_kimg * 1e3), 0) * self.blur_init_sigma if self.blur_fade_kimg > 0 else 0
        r1_gamma = self.r1_gamma

        alpha = min(cur_nimg / (self.gpc_reg_fade_kimg * 1e3), 1) if self.gpc_reg_fade_kimg > 0 else 1
        swapping_prob = (1 - alpha) * 1 + alpha * self.gpc_reg_prob if self.gpc_reg_prob is not None else None

        if self.neural_rendering_resolution_final is not None:
            alpha = min(cur_nimg / (self.neural_rendering_resolution_fade_kimg * 1e3), 1)
            neural_rendering_resolution = int(np.rint(self.neural_rendering_resolution_initial * (1 - alpha) + self.neural_rendering_resolution_final * alpha))
        else:
            neural_rendering_resolution = self.neural_rendering_resolution_initial

        real_img_raw = filtered_resizing(real_img, size=neural_rendering_resolution, f=self.resample_filter, filter_mode=self.filter_mode)

        if self.blur_raw_target:
            blur_size = np.floor(blur_sigma * 3)
            if blur_size > 0:
                f = torch.arange(-blur_size, blur_size + 1, device=real_img_raw.device).div(blur_sigma).square().neg().exp2()
                real_img_raw = upfirdn2d.filter2d(real_img_raw, f / f.sum())

        real_img = {'image': real_img, 'image_raw': real_img_raw}

        gs_params = {}
        
        if self.coeffs['is_resume']:
            gs_params['disable_background'] = self.coeffs['disable_background']
        else:
            gs_params['disable_background'] = True if cur_nimg < 5e+4 else self.coeffs['disable_background'] # initialize background after 5e+4 images for stability
        

        # Gmain: Maximize logits for generated images.
        if phase in ['Gmain', 'Gboth']:
            with torch.autograd.profiler.record_function('Gmain_forward'):
                
                # For FFHQ, augment pose distribution (Appendix A.1)
                gen_c = self.sample_custom_camera(gen_c, p=self.coeffs['prob_uniform'], horizontal_std=self.coeffs['horizontal_std'], vertical_std=self.coeffs['vertical_std'], \
                                                    htype=self.coeffs['horizontal_type'], vtype=self.coeffs['vertical_type'])
                                                    
                
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, gs_params=gs_params)
                
                gen_logits, aux_out = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, return_aux=True)
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                loss_Gmain = torch.nn.functional.softplus(-gen_logits)
                training_stats.report('Loss/G/loss', loss_Gmain)
                
                sampled_coords = gen_img['sample_coordinates']
                anchors = gen_img['anchors'][:, :self.G.num_pts]
                dist = knn_distance(anchors[:, :], k=self.coeffs['knn_num_ks'])
                dist_center = anchors.mean(dim=1).square().mean() # (avg(xyz) - 0).square 
                
                
                feat_contrastive = aux_out["feat_contrastive"]
                camera_contrastive = aux_out["c_contrastive"]
                if self.coeffs['contrastive_negpose_mult'] > 0:
                    camera_contrastive_negs = self.D.aux_branch_camera(self.eg3d_to_3dgs(self.sample_gaussian_camera(c=gen_c.repeat([self.coeffs['contrastive_negpose_mult'], 1]), p=1.0)))
                else:
                    camera_contrastive_negs = camera_contrastive[:0]
                
                # Multi-gpu
                feat_contrastive = torch.cat(GatherLayer.apply(feat_contrastive), dim=0)
                camera_contrastive = torch.cat(GatherLayer.apply(camera_contrastive), dim=0)
                camera_contrastive_negs = torch.cat(GatherLayer.apply(camera_contrastive_negs), dim=0)
            
                G_contrastive = contrastive_loss(feat_contrastive, torch.cat([camera_contrastive, camera_contrastive_negs], dim=0))
                training_stats.report('Loss/G_contrastive', G_contrastive.mean().item())

                
            with torch.autograd.profiler.record_function('Gmain_backward'):
                ((loss_Gmain + G_contrastive * self.coeffs['pose_contrastive']).mean().mul(gain) + \
                        dist_center.mean() * self.coeffs['center_dists'] + \
                        dist.mean() * self.coeffs['knn_dists']).backward()

                # Check param
                G_norm_param = compute_param_norm(self.G.parameters())
                G_norm_grad = clip_grad_norm_(self.G.parameters(), max_norm=20)

             
            
        # Dmain: Minimize logits for generated images.
        loss_Dgen = 0
        if phase in ['Dmain', 'Dboth']:
            with torch.autograd.profiler.record_function('Dgen_forward'):
                
                # For FFHQ, augment pose distribution (Appendix A.1)
                gen_c = self.sample_custom_camera(gen_c, p=self.coeffs['prob_uniform'], horizontal_std=self.coeffs['horizontal_std'], vertical_std=self.coeffs['vertical_std'], \
                                                    htype=self.coeffs['horizontal_type'], vtype=self.coeffs['vertical_type'])
                
                gen_img, _gen_ws = self.run_G(gen_z, gen_c, swapping_prob=swapping_prob, neural_rendering_resolution=neural_rendering_resolution, update_emas=True, gs_params=gs_params)
                
                gen_logits, aux_out = self.run_D(gen_img, gen_c, blur_sigma=blur_sigma, update_emas=True, return_aux=True)
                
                training_stats.report('Loss/scores/fake', gen_logits)
                training_stats.report('Loss/signs/fake', gen_logits.sign())
                
                    
                feat_contrastive = aux_out["feat_contrastive"]
                camera_contrastive = aux_out["c_contrastive"]
                if self.coeffs['contrastive_negpose_mult'] > 0:
                    camera_contrastive_negs = self.D.aux_branch_camera(self.eg3d_to_3dgs( \
                                self.sample_gaussian_camera(c=gen_c.repeat([self.coeffs['contrastive_negpose_mult'], 1]), p=1.0))
                                )
                else:
                    camera_contrastive_negs = camera_contrastive[:0]
                
                # Multi-gpu
                feat_contrastive = torch.cat(GatherLayer.apply(feat_contrastive), dim=0)
                camera_contrastive = torch.cat(GatherLayer.apply(camera_contrastive), dim=0)
                camera_contrastive_negs = torch.cat(GatherLayer.apply(camera_contrastive_negs), dim=0)
                D_gen_contrastive = contrastive_loss(feat_contrastive, torch.cat([camera_contrastive, camera_contrastive_negs], dim=0))
                training_stats.report('Loss/D_gen_contrastive', D_gen_contrastive.mean().item())

                loss_Dgen = torch.nn.functional.softplus(gen_logits)
                
            with torch.autograd.profiler.record_function('Dgen_backward'):
                (loss_Dgen + D_gen_contrastive * self.coeffs['pose_contrastive'] * 0.).mean().mul(gain).backward() # Do not use contrastive loss for D_gen
                
                # Check parameter norm
                D_norm_param = compute_param_norm(self.D.parameters())
                D_norm_grad = clip_grad_norm_(self.D.parameters(), max_norm=5)
                    

        # Dmain: Maximize logits for real images.
        # Dr1: Apply R1 regularization.
        if phase in ['Dmain', 'Dreg', 'Dboth']:
            name = 'Dreal' if phase == 'Dmain' else 'Dr1' if phase == 'Dreg' else 'Dreal_Dr1'
            with torch.autograd.profiler.record_function(name + '_forward'):
                real_img_tmp_image = real_img['image'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp_image_raw = real_img['image_raw'].detach().requires_grad_(phase in ['Dreg', 'Dboth'])
                real_img_tmp = {'image': real_img_tmp_image, 'image_raw': real_img_tmp_image_raw}

                real_logits, aux_out = self.run_D(real_img_tmp, real_c, blur_sigma=blur_sigma, return_aux=True)
                training_stats.report('Loss/scores/real', real_logits)
                training_stats.report('Loss/signs/real', real_logits.sign())

                # Pose contrastive Loss
                feat_contrastive = aux_out["feat_contrastive"]
                camera_contrastive = aux_out["c_contrastive"]
                if self.coeffs['contrastive_negpose_mult'] > 0:
                    camera_contrastive_negs = self.D.aux_branch_camera(self.eg3d_to_3dgs( \
                        self.sample_gaussian_camera(c=real_c.repeat([self.coeffs['contrastive_negpose_mult'], 1]), p=1.0))
                        )
                else:
                    camera_contrastive_negs = camera_contrastive[:0]
                
                # Multi-gpu
                feat_contrastive = torch.cat(GatherLayer.apply(feat_contrastive), dim=0)
                camera_contrastive = torch.cat(GatherLayer.apply(camera_contrastive), dim=0)
                camera_contrastive_negs = torch.cat(GatherLayer.apply(camera_contrastive_negs), dim=0)
                D_real_contrastive = contrastive_loss(feat_contrastive, torch.cat([camera_contrastive, camera_contrastive_negs], dim=0))
                training_stats.report('Loss/D_real_contrastive', D_real_contrastive.mean().item())
                
                loss_Dreal = 0
                if phase in ['Dmain', 'Dboth']:
                    loss_Dreal = torch.nn.functional.softplus(-real_logits)
                    training_stats.report('Loss/D/loss', loss_Dgen + loss_Dreal)

                loss_Dr1 = 0
                if phase in ['Dreg', 'Dboth']:
                    if self.dual_discrimination:
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image'], real_img_tmp['image_raw']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                            r1_grads_image_raw = r1_grads[1]
                        r1_penalty = r1_grads_image.square().sum([1,2,3]) + r1_grads_image_raw.square().sum([1,2,3])
                        
                    else: # single discrimination
                        with torch.autograd.profiler.record_function('r1_grads'), conv2d_gradfix.no_weight_gradients():
                            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp['image']], create_graph=True, only_inputs=True)
                            r1_grads_image = r1_grads[0]
                        r1_penalty = r1_grads_image.square().sum([1,2,3])
                    loss_Dr1 = r1_penalty * (r1_gamma / 2)
                    training_stats.report('Loss/r1_penalty', r1_penalty)
                    training_stats.report('Loss/D/reg', loss_Dr1)

            with torch.autograd.profiler.record_function(name + '_backward'):
                (loss_Dreal + loss_Dr1 + D_real_contrastive * self.coeffs['pose_contrastive']).mean().mul(gain).backward()
                
                # clip gradient norm
                D_norm_grad = clip_grad_norm_(self.D.parameters(), max_norm=5)
              
              

#----------------------------------------------------------------------------
# Belows are the loss functions for KNN distance and contrastive loss

from typing import Any, Dict, List, Union, Iterable
from numbers import Number
import torch
import torch.distributed as dist
PARAMETERS_DTYPE = Union[torch.Tensor, Iterable[torch.Tensor]]

def knn_distance(pos, k, num=-1, idx=None, return_idx=False):
    """
    Args:
        pos: position point cloud [B, N, 3]
        k: kNN neighbours
    Return:
        [B, 2dims, N, k]    
    """
    x = pos.permute(0, 2, 1)
    B, dims, N = x.shape

    # batched pair-wise distance
    if idx is None:
        xt = x.permute(0, 2, 1)
        xi = -2 * torch.bmm(xt, x)
        xs = torch.sum(xt**2, dim=2, keepdim=True)
        xst = xs.permute(0, 2, 1)
        dist = xi + xs + xst # [B, N, N]

        # get k NN id
        _, idx_o = torch.sort(dist, dim=2)
        idx = idx_o[: ,: ,1:k+1] # [B, N, k]
        idx = idx.contiguous().view(B, N*k)


    # gather
    neighbors = []
    for b in range(B):
        tmp = torch.index_select(x[b], 1, idx[b]) # [d, N*k] <- [d, N], 0, [N*k]
        tmp = tmp.view(dims, N, k)
        neighbors.append(tmp)

    neighbors = torch.stack(neighbors) # [B, d, N, k]

    # centralize
    central = x.unsqueeze(3) # [B, d, N, 1]
    central = central.repeat(1, 1, 1, k) # [B, d, N, k]

    return (central - neighbors).square().mean(dim=[1, 3])


def contrastive_loss_sym(x1, x2=None, temperature=0.1):
    labels = torch.arange(x1.shape[0])
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(x1.device)
    
    if x2 is None:
        x2 = x1
        remove_identity_mx = True
    else:
        remove_identity_mx = False
        
    features1 = torch.nn.functional.normalize(x1, dim=-1) # channel dim
    features2 = torch.nn.functional.normalize(x2, dim=-1)
    similarity_matrix = torch.matmul(features1, features2.T)
        
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features1.device)
    
    if remove_identity_mx:
        # if x2 is none, remove identical datapoints else use all
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    else:
        labels = labels.view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix.view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(x1.device)


    logits = logits / temperature
    # return logits, labels
    
    return torch.nn.functional.cross_entropy(logits, labels)


def contrastive_loss(x1, x2=None, temperature=0.1):
    labels = (torch.arange(x2.shape[0]).unsqueeze(0) == torch.arange(x1.shape[0]).unsqueeze(1)).float()
    labels = labels.to(x1.device)
    
    if x2 is None:
        x2 = x1
        remove_identity_mx = True
    else:
        remove_identity_mx = False
        
    features1 = torch.nn.functional.normalize(x1, dim=-1) # channel dim
    features2 = torch.nn.functional.normalize(x2, dim=-1)
    similarity_matrix = torch.matmul(features1, features2.T)

    mask = labels.to(dtype=torch.bool)
    
    if remove_identity_mx:
        # if x2 is none, remove identical datapoints else use all
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    else:
        labels = labels.view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix.view(similarity_matrix.shape[0], -1)

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long).to(x1.device)


    logits = logits / temperature
    # return logits, labels
    
    return torch.nn.functional.cross_entropy(logits, labels)


def pose_matching_loss(x1, x2=None, x2_neg=None):

    # shuffle x2
    if x2_neg is None:
        x2_shuffled = x2[torch.randperm(x2.shape[0])]
        neg_pose_matchness = torch.nn.functional.softplus(x1 * x2_shuffled)
    else:
        neg_pose_matchness = torch.nn.functional.softplus(x1 * x2_neg)
    
    pos_pose_matchness = torch.nn.functional.softplus(-1 * x1 * x2)

    return pos_pose_matchness.mean() + neg_pose_matchness.mean()



def all_gather_tensor(tensor: torch.Tensor) -> List[torch.Tensor]:
    if dist.is_initialized() and dist.is_available():
        world_size = dist.get_world_size()
        local_rank = dist.get_rank()
        output = [
            tensor if (i == local_rank) else torch.empty_like(tensor) for i in range(world_size)
        ]
        dist.all_gather(output, tensor, async_op=False)
        return output
        # return torch.cat(output, dim=0)
    else:
        return [tensor]
    
    
# https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/gather.py
class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
    
    
def gather_for_contrastive(x):
    return torch.cat(GatherLayer.apply(x), dim=0)


@torch.no_grad()
def compute_param_norm(parameters: PARAMETERS_DTYPE, norm_type: float = 2.0,
                       requires_grad: bool = True) -> torch.Tensor:
    """Compute parameter norm.

    Args:
        parameters:             iterable of parameters (List, Tuple, Iter, ...)
        norm_type (float):      default l2 norm (2.0)
        requires_grad (bool):   whether to count only parameters with requires_grad=True.
    Returns:
        Tensor:              (1,) scalar
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    if requires_grad:
        parameters = [p for p in parameters if p.requires_grad]
    else:
        parameters = list(parameters)
    if len(parameters) == 0:
        return torch.as_tensor(0., dtype=torch.float32)

    device = parameters[0].device
    total_norm = torch.norm(torch.stack([torch.norm(p, norm_type).to(device) for p in parameters]), norm_type)
    return total_norm
