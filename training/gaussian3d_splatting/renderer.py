import os
import math
import numpy as np
from typing import NamedTuple
from plyfile import PlyData, PlyElement

import torch
from torch import nn

from diff_gaussian_rasterization import (
    GaussianRasterizationSettings,
    GaussianRasterizer,
)
# from simple_knn._C import distCUDA2

from .sh_utils import eval_sh, SH2RGB, RGB2SH
# from .mesh import Mesh
# from .mesh_utils import decimate_mesh, clean_mesh

from .bg_utils import rand_fft_image, image_sample
from torch_utils.ops import upfirdn2d

# import kiui

def inverse_sigmoid(x):
    return torch.log(x/(1-x))

def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    
    def helper(step):
        if lr_init == lr_final:
            # constant lr, ignore other params
            return lr_init
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def strip_lowerdiag(L):
    # uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device=L.device)

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty

def strip_symmetric(sym):
    return strip_lowerdiag(sym)

def gaussian_3d_coeff(xyzs, covs):
    # xyzs: [N, 3]
    # covs: [N, 6]
    x, y, z = xyzs[:, 0], xyzs[:, 1], xyzs[:, 2]
    a, b, c, d, e, f = covs[:, 0], covs[:, 1], covs[:, 2], covs[:, 3], covs[:, 4], covs[:, 5]

    # eps must be small enough !!!
    inv_det = 1 / (a * d * f + 2 * e * c * b - e**2 * a - c**2 * d - b**2 * f + 1e-24)
    inv_a = (d * f - e**2) * inv_det
    inv_b = (e * c - b * f) * inv_det
    inv_c = (e * b - c * d) * inv_det
    inv_d = (a * f - c**2) * inv_det
    inv_e = (b * c - e * a) * inv_det
    inv_f = (a * d - b**2) * inv_det

    power = -0.5 * (x**2 * inv_a + y**2 * inv_d + z**2 * inv_f) - x * y * inv_b - x * z * inv_c - y * z * inv_e

    power[power > 0] = -1e10 # abnormal values... make weights 0
        
    return torch.exp(power)

def build_rotation(r):
    norm = torch.sqrt(r[:,0]*r[:,0] + r[:,1]*r[:,1] + r[:,2]*r[:,2] + r[:,3]*r[:,3])

    q = r / norm[:, None]

    # R = torch.zeros((q.size(0), 3, 3), device='cuda')
    R = torch.zeros((q.size(0), 3, 3), device=r.device)

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

def build_scaling_rotation(s, r):
    # L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=r.device)
    R = build_rotation(r)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    return L

class BasicPointCloud(NamedTuple):
    points: np.array
    colors: np.array
    normals: np.array

def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    symm = strip_symmetric(actual_covariance)
    return symm

def build_covariance_from_scaling_rotation_cov(scaling, scaling_modifier, rotation):
    L = build_scaling_rotation(scaling_modifier * scaling, rotation)
    actual_covariance = L @ L.transpose(1, 2)
    return actual_covariance

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    P = torch.zeros(4, 4)

    z_sign = 1.0
    
    if tanHalfFovX == 0 or tanHalfFovY == 0:
        # dummy inputs are given
        P[0, 0] = 1
        P[1, 1] = 1
    else:
        P[0, 0] = 1 / tanHalfFovX
        P[1, 1] = 1 / tanHalfFovY
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


class MiniCam:
    def __init__(self, c2w, width, height, fovy, fovx, znear, zfar, device=None):
        # c2w (pose) should be in NeRF convention.

        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar

        try:
            w2c = torch.linalg.inv(c2w)
        except:
            w2c = c2w.clone() # when dummy c2w are given
        
        # rectify...
        w2c[1:3, :3] *= -1
        w2c[:3, 3] *= -1

        self.world_view_transform = w2c.clone().transpose(0, 1)
        self.projection_matrix = (
            getProjectionMatrix(
                znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy
            )
            .transpose(0, 1)
        ).to(w2c.device)
        self.full_proj_transform = self.world_view_transform @ self.projection_matrix
        self.camera_center = - c2w[:3, 3].clone()

        if device is not None:
            self.world_view_transform = self.world_view_transform.to(device)
            self.projection_matrix = self.projection_matrix.to(device)
            self.full_proj_transform = self.full_proj_transform.to(device)
            self.camera_center = self.camera_center.to(device)
            


class Renderer:
    def __init__(self, sh_degree=3, white_background=False, radius=1):
        
        self.sh_degree = sh_degree
        self.white_background = white_background
        self.radius = radius

        # self.gaussians = GaussianModel(sh_degree)

        self.bg_color = torch.tensor(
            [1, 1, 1] if white_background else [0, 0, 0],
            dtype=torch.float32,
        )
                
        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        
        # init param?
        self.active_sh_degree = 0
        self.max_sh_degree = 0
        
    def get_scaling(self, _scaling):
        return self.scaling_activation(_scaling)
    
    def get_rotation(self, _rotation):
        return self.rotation_activation(_rotation)
    
    def get_xyz(self):
        return self._xyz
    
    def get_features(self, features_dc, features_rest):
        return torch.cat((features_dc, features_rest), dim=1)
    
    def get_opacity(self, _opacity):
        return self.opacity_activation(_opacity)
    
    
    
    def render(
        self,
        gaussian_params,
        viewpoint_camera,
        scaling_modifier=1.0,
        invert_bg_color=False,
        override_color=None,
        compute_cov3D_python=False,
        convert_SHs_python=False,
        random_background=True,
    ):
        
        _xyz = gaussian_params["_xyz"]
        _features_dc = gaussian_params["_features_dc"]
        _features_rest = gaussian_params["_features_rest"]
        _scaling = gaussian_params["_scaling"]
        _rotation = gaussian_params["_rotation"]
        _opacity = gaussian_params["_opacity"]
        
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                # self.gaussians.get_xyz,
                # dtype=self.gaussians.get_xyz.dtype,
                _xyz,
                dtype=_xyz.dtype,
                requires_grad=True,
                device=_xyz.device
            )
            + 0
        )
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
            bg=self.bg_color.to(_xyz.device) if not invert_bg_color else 1 - self.bg_color.to(_xyz.device),
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform.to(_xyz.device),
            projmatrix=viewpoint_camera.full_proj_transform.to(_xyz.device),
            sh_degree=self.active_sh_degree,
            campos=viewpoint_camera.camera_center.to(_xyz.device),
            prefiltered=False,
            debug=False,
            # debug=True,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        rasterizer = rasterizer.to(_xyz.device)
        

        # means3D = self.gaussians.get_xyz
        means3D = _xyz
        means2D = screenspace_points
        # opacity = self.gaussians.get_opacity
        opacity = self.get_opacity(_opacity)
        

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if compute_cov3D_python:
            # cov3D_precomp = self.gaussians.get_covariance(scaling_modifier)
            cov3D_precomp = self.get_covariance(scaling_modifier)
        else:
            # scales = self.gaussians.get_scaling
            # rotations = self.gaussians.get_rotation
            scales = self.get_scaling(_scaling)
            rotations = self.get_rotation(_rotation)
            
        # check nan
        if torch.isnan(scales).sum() > 0.:
            # print(torch.isnan(_scaling).sum(), _scaling[torch.where(torch.isnan(_scaling))])
            print("# nans in {}".format(scales.device), torch.isnan(scales).sum().item())
        # scales[torch.where(torch.isnan(scales))] = 0.0
        scales = torch.nan_to_num(scales)
        
        # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
        # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
        shs = None
        colors_precomp = None
        if colors_precomp is None:
            if convert_SHs_python:
                pass
                # shs_view = self.gaussians.get_features.transpose(1, 2).view(
                #     -1, 3, (self.gaussians.max_sh_degree + 1) ** 2
                # )
                # dir_pp = self.gaussians.get_xyz - viewpoint_camera.camera_center.repeat(
                #     self.gaussians.get_features.shape[0], 1
                # )
                # dir_pp_normalized = dir_pp / dir_pp.norm(dim=1, keepdim=True)
                # sh2rgb = eval_sh(
                #     self.gaussians.active_sh_degree, shs_view, dir_pp_normalized
                # )
                # colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            else:
                # shs = self.gaussians.get_features
                shs = self.get_features(_features_dc, _features_rest)
        else:
            colors_precomp = override_color

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        with torch.autocast(device_type=_xyz.device.type, dtype=torch.float32):
            # rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
            rendered_image, radii, rendered_depth, rendered_alpha = rasterizer(
                means3D=means3D,
                means2D=means2D,
                shs=shs,
                colors_precomp=colors_precomp,
                opacities=opacity,
                scales=scales,
                rotations=rotations,
                cov3D_precomp=cov3D_precomp,
            )
        
        
        # rendered_image = rendered_image.clamp(0, 1)
        rendered_image = rendered_image / 0.5 - 1.
        # rendered_image = torch.tanh(rendered_image)

        zero_idx = torch.where(rendered_depth == 0, 1., 0.)
        rendered_depth = rendered_depth * (1. - zero_idx) + torch.ones_like(rendered_depth) * torch.max(rendered_depth) * zero_idx

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        # They will be excluded from value updates used in the splitting criteria.
        return {
            "image": rendered_image,
            "depth": rendered_depth,
            "alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter": radii > 0,
            "radii": radii,
        }
