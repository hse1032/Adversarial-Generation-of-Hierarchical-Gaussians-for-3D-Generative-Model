import torch
import torch.nn as nn
import numpy as np
import math
from itertools import islice

from training.networks_stylegan2 import FullyConnectedLayer, Conv2dLayer, SynthesisLayer
from training.transformer import Transformer, MLP, AdaptiveNorm

from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import bias_act
import torch.nn.init as init
from training.gaussian3d_splatting.sh_utils import RGB2SH

import torch.nn.functional as F

@persistence.persistent_class
class ModulatedFullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        w_dim,                      # Number of w features
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
        weight_init     = 1,
        clamp           = 256,
        residual        = False,    # Use residual block?
    ):
        super().__init__()
        
        # Only adding modulation (not demodulation)
        
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier * weight_init)
        self.bias = torch.nn.Parameter(torch.full([out_features], np.float32(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier
        
        self.clamp = clamp
        self.affine = FullyConnectedLayer(w_dim, in_features, bias_init=1)
        
        self.residual = residual
        if residual:
            if in_features != out_features:
                self.res_fc = FullyConnectedLayer(in_features, out_features, activation='linear')
            else:
                self.res_fc = Identity()
            self.layerscale = torch.nn.Parameter(torch.ones([1, 1, out_features]) * 1e-2)

    def forward(self, x, w, demodulate=True):
        # x: [B, L, C] / w: [B, w_dim]
        
        B, L, C = x.shape
        x_original = x
        
        styles = self.affine(w) # [B, C]
        weight = self.weight
        
        # Pre-normalize inputs to avoid FP16 overflow.
        if x.dtype == torch.float16 and demodulate:
            weight = weight * (1 / np.sqrt(C) / weight.norm(float('inf'), dim=[1], keepdim=True)) # max_Ikk
            styles = styles / styles.norm(float('inf'), dim=-1, keepdim=True) # max_I
        
        weight_styled = weight.unsqueeze(0) # [NOI]
        weight_styled = weight_styled * styles.reshape(B, 1, -1) # [NOI]
        if demodulate:
            dcoefs = (weight_styled.square().sum(dim=[2]) + 1e-8).rsqrt() # [NO]
        
        if not demodulate:
            styles = styles * self.weight_gain
            
        x = x * styles.to(x.dtype).unsqueeze(1)
        x = x.reshape(-1, C) # [B * L, C]
        
        w = weight.to(x.dtype) 

        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        x = x.matmul(w.t())
        if demodulate:
            x = x * (dcoefs.to(x.dtype).repeat_interleave(L, dim=0))
        
        act_clamp = self.clamp * 1. if self.clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, clamp=act_clamp)
        
        x = x.reshape(B, L, -1)
        
        if self.residual:
            x = x * self.layerscale.to(x.dtype) + self.res_fc(x_original)
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, out_features={self.out_features:d}, activation={self.activation:s}'

@persistence.persistent_class
class Identity(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
    
    def forward(self, x, **kwargs):
        # do nothing
        return x


@persistence.persistent_class
class AdaMLP(nn.Module):
    def __init__(
        self,
        width,                      # Width of the network.
        adaptive_layerscale=False
    ):
        super().__init__()
        
        self.norm = AdaptiveNorm(width)
        self.mlp = MLP(width=width, init_scale=None)
        self.ls = FullyConnectedLayer(512, width, activation='linear', weight_init=0.)
    
    def forward(self, x, w):
        return x + self.mlp(self.norm(x, w)) * self.ls(w).unsqueeze(1)
    

@persistence.persistent_class
class PlainMLP(nn.Module):
    def __init__(
        self,
        width,                      # Width of the network.
    ):
        super().__init__()
        
        self.mlp = MLP(width=width, init_scale=None)
        self.ls = torch.nn.Parameter(torch.ones(size=[width]) * 0)
    
    def forward(self, x, w=None):
        return x + self.mlp(x) * self.ls


@persistence.persistent_class
class LinLinear(nn.Module):
    def __init__(self, ch_in, ch_out, is_first=False, bias=True):
        super().__init__()
        
        self.linear = nn.Linear(ch_in, ch_out, bias=bias)
        if is_first:
            nn.init.uniform_(self.linear.weight, -np.sqrt(9 / ch_in), np.sqrt(9 / ch_in))
        else:
            nn.init.uniform_(self.linear.weight, -np.sqrt(3 / ch_in), np.sqrt(3 / ch_in))
    
    def forward(self, x):
        return self.linear(x)


@persistence.persistent_class
class SinActivation(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x):
        return torch.sin(x)


@persistence.persistent_class
class LFF(nn.Module):
    def __init__(self, hidden_size, ):
        super().__init__()
        self.ffm = LinLinear(3, hidden_size, is_first=True)
        self.activation = SinActivation()

    def forward(self, x):
        x = self.ffm(x)
        x = self.activation(x)
        return x

@persistence.persistent_class
class constant_PE(nn.Module):
    def __init__(self, pe_dim, pe_res):
        super().__init__()

        self.pe_dim = pe_dim

        # constant PE
        self.const_pe = torch.nn.Parameter(torch.randn([3, pe_dim, 1, pe_res]))
        
    def forward(self, pos):
        B, L, _ = pos.shape
        
        x_coord = pos # (x + 1) / 2 # range of x: (-1, 1)
        x_coord = x_coord.unsqueeze(-1) # [B, L, 3, 1]
        x_coord = x_coord.permute(0, 2, 1, 3).reshape(3 * B, 1, L, 1) # [B * 3, 1, L, 1]
        x_coord = torch.cat([torch.zeros_like(x_coord), x_coord], dim=-1) # [B * 3, 1, L, 2]
        
        const_pe = self.const_pe.repeat([B, 1, 1, 1]) # [B * 3, C, 1, L]
        const_emb = torch.nn.functional.grid_sample(const_pe, x_coord, mode='bilinear') # [B * 3, C, 1, L]
        const_emb = const_emb.reshape(B, 3, self.pe_dim, 1, L).sum(1).reshape(B, self.pe_dim, L).permute(0, 2, 1) # [B, L, C]
        return const_emb / np.sqrt(3)
        
        
@persistence.persistent_class
class PointUpsample_subpixel(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        upsample_ratio,
        resolution,
        pe_dim = 96,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.upsample_ratio = upsample_ratio
        
        self.use_pe = pe_dim > 0
        
        if self.use_pe: 
            self.coord_injection = CoordInjection_const(pe_dim, resolution)
            self.subpixel = FullyConnectedLayer(in_features + pe_dim * 2, out_features * (upsample_ratio), activation='lrelu')
        else:
            self.subpixel = FullyConnectedLayer(in_features, out_features * (upsample_ratio), activation='lrelu')
            
        if in_features != out_features:
            self.res_fc = FullyConnectedLayer(in_features, out_features, activation='linear')
        else:
            self.res_fc = Identity()

    def forward(self, x, pc, w=None, pe_data=None):
        # x: [B, L, C], 
        # pc: [xyz, scale, rotation, color, opacity]
        B, L, C = x.shape
        
        if self.use_pe:
            if pe_data is None:
                x_pe = self.coord_injection(pc[0], x)
            else:
                x_pe = self.coord_injection(pe_data, x)
        else:
            x_pe = x
        x_upsampled = self.subpixel(x_pe).reshape(B, L * self.upsample_ratio, self.out_features)
        x_upsampled = (self.res_fc(x).repeat_interleave(self.upsample_ratio, dim=1) + x_upsampled) / np.sqrt(2)
        
        if pc is not None and self.upsample_ratio > 1:
            # # No split and clone
            pc_upsampled = [_pc.repeat_interleave(self.upsample_ratio, dim=1) for _pc in pc] # [B, L * r, C]
        else:
            pc_upsampled = pc
    
        return x_upsampled, pc_upsampled
    
    

@persistence.persistent_class
class PointUpsample_modsubpixel(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        upsample_ratio,
        resolution,
        w_dim=512,
        pe_dim = 96,
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.upsample_ratio = upsample_ratio
        
        self.use_pe = pe_dim > 0
        
        if self.use_pe: 
            self.coord_injection = CoordInjection_const(pe_dim, resolution)
            self.subpixel = ModulatedFullyConnectedLayer(in_features + pe_dim * 2, out_features * (upsample_ratio), w_dim, activation='lrelu')
        else:
            self.subpixel = ModulatedFullyConnectedLayer(in_features + pe_dim * 2, out_features * (upsample_ratio), w_dim, activation='lrelu')
            
        if in_features != out_features:
            self.res_fc = FullyConnectedLayer(in_features, out_features, activation='linear')
        else:
            self.res_fc = Identity()

    def forward(self, x, pc, w, pe_data=None):
        # x: [B, L, C], 
        # pc: [xyz, scale, rotation, color, opacity]
        B, L, C = x.shape
        
        if self.use_pe:
            if pe_data is None:
                x_pe = self.coord_injection(pc[0], x)
            else:
                x_pe = self.coord_injection(pe_data, x)
        else:
            x_pe = x
        x_upsampled = self.subpixel(x_pe, w).reshape(B, L * self.upsample_ratio, self.out_features)
        x_upsampled = (self.res_fc(x).repeat_interleave(self.upsample_ratio, dim=1) + x_upsampled) / np.sqrt(2)
        
        if pc is not None and self.upsample_ratio > 1:
            # # No split and clone
            pc_upsampled = [_pc.repeat_interleave(self.upsample_ratio, dim=1) for _pc in pc] # [B, L * r, C]
        else:
            pc_upsampled = pc
    
        return x_upsampled, pc_upsampled
        
@persistence.persistent_class
class CoordInjection_const(torch.nn.Module):
    def __init__(self,
        pe_dim,                 # Number of input features.
        pe_res,                 # resolution for constant pe
    ):
        super().__init__()
        
        self.learnable_pe = LFF(pe_dim)

        # constant PE
        self.const_pe = constant_PE(pe_dim, int(np.sqrt(pe_res)))
            

    def forward(self, pos, x=None, type='cat'):
        if x is not None:
            x = torch.cat([x, self.learnable_pe(pos).to(x.dtype), self.const_pe(pos).to(x.dtype)], dim=-1)
        else:
            x = torch.cat([self.learnable_pe(pos), self.const_pe(pos)], dim=-1)
        return x
        
        
def get_scaled_directional_vector_from_quaternion(r, s):
    # r, s: [B, npoints, c]
    N, npoints, _ = r.shape
    r, s = r.reshape([N * npoints, -1]), s.reshape([N * npoints, -1])
    
    # Rotation activation (normalize)
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
    
    # Scaling activation (exp)
    s = torch.exp(s)
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device=r.device)

    L[:,0,0] = s[:,0]
    L[:,1,1] = s[:,1]
    L[:,2,2] = s[:,2]

    L = R @ L
    
    L = L.reshape([N, npoints, 3, 3])
    return L

def get_edge_features(x, k, num=-1, idx=None, return_idx=False):
    """
    Args:
        x: point cloud [B, dims, N]
        k: kNN neighbours
    Return:
        [B, 2dims, N, k]    
    """
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

    ee = torch.cat([central, neighbors-central], dim=1)
    assert ee.shape == (B, 2*dims, N, k)

    if return_idx:
        return ee, idx
    return ee


def inverse_sigmoid(x):
    return torch.log(x/(1-x))


@persistence.persistent_class
class AdaEdgeBlock(nn.Module):
    # changed from Edgeblock in SPGAN
    def __init__(self, Fin, Fout, k):
        super().__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        
        self.norm = AdaptiveNorm(Fin)
        
        self.conv_w = nn.Sequential(
            Conv2dLayer(Fin, Fout//2, kernel_size=1, activation='lrelu'),
            Conv2dLayer(Fout//2, Fout, kernel_size=1, activation='linear'),
        )

        self.conv_x = nn.Sequential(
            Conv2dLayer(2 * Fin, Fout, kernel_size=1, activation='lrelu'),
            Conv2dLayer(Fout, Fout, kernel_size=1, activation='linear'),
        )

        self.ls = FullyConnectedLayer(512, Fout, activation='linear', weight_init=0.)

    def forward(self, x, w=None, pos=None, upsample_factor=None):
        # x: [B, L, C]
        x_original = x
        
        x = self.norm(x, w)
        
        x = x.permute(0, 2, 1) # [B, C, L]
        
        B, C, N = x.shape
        x = get_edge_features(x, self.k) # [B, 2Fin, N, k]
        
        w_softmax = self.conv_w(x[:, C:, :, :])
        
        w_softmax = F.softmax(w_softmax.to(torch.float32), dim=-1)  # [B, Fout, N, k] -> [B, Fout, N, k]
        w_softmax = w_softmax.to(x.dtype)

        x = self.conv_x(x)  # Bx2CxNxk
        
        x = (x * w_softmax).sum(-1, keepdim=True)

        x = x.squeeze(3)  # BxCxN
        
        return x.permute(0, 2, 1) * self.ls(w).unsqueeze(1) + x_original


@persistence.persistent_class
class LocalEdgeBlock(nn.Module):
    # changed from Edgeblock in SPGAN
    def __init__(self, Fin, Fout):
        super().__init__()
        self.Fin = Fin
        self.Fout = Fout
        
        self.conv_w = nn.Sequential(
            Conv2dLayer(Fin, Fout//2, kernel_size=1, activation='lrelu'),
            Conv2dLayer(Fout//2, Fout, kernel_size=1, activation='linear'),
        )

        self.conv_x = nn.Sequential(
            Conv2dLayer(2 * Fin, Fout, kernel_size=1, activation='lrelu'),
            Conv2dLayer(Fout, Fout, kernel_size=1, activation='linear'),
        )
        
        # residual?
        self.ls = torch.nn.Parameter(torch.ones([Fout]) * 0.0)

    def forward(self, x, w=None, pos=None, upsample_factor=None):
        upsample_factor = 4
        
        # x: [B, L, C]
        x_original = x
        
        _B, _L, _C = x.shape
        x = x.reshape(_B, _L // upsample_factor, upsample_factor, _C)
        x_center = x.unsqueeze(3).repeat([1, 1, 1, upsample_factor, 1]).reshape(_B, _L, upsample_factor, _C)
        x_delta = (x.unsqueeze(2) - x.unsqueeze(3)).reshape(_B, _L, upsample_factor, _C) # [B, L, 1, C]
        
        x_center = x_center.permute(0, 3, 1, 2) # [B, C, L, ups]
        x_delta = x_delta.permute(0, 3, 1, 2) # [B, C, L, ups]
        
        B, C, L, k = x_center.shape
        x = torch.cat([x_center, x_delta], dim=1) # [B, 2C, L, ups]
        
        w = self.conv_w(x[:, C:, :, :])
        w = F.softmax(w.to(torch.float32), dim=-1)  # [B, Fout, N, k] -> [B, Fout, N, k]
        w = w.to(x.dtype)

        x = self.conv_x(x)  # Bx2CxNxk
        
        x = (x * w).sum(-1, keepdim=True)
        x = x.squeeze(3)  # BxCxN
        
        return x.permute(0, 2, 1) * self.ls + x_original
    

@persistence.persistent_class
class PointGenerator(nn.Module):
    def __init__(self,
            w_dim,                      # Intermediate latent (W) dimensionality.
            img_resolution,             # Output image resolution.
            img_channels,               # Number of color channels.
            channel_base    = 32768,    # Overall multiplier for the number of channels.
            channel_max     = 512,      # Maximum number of channels in any layer.
            knn=16,
            init_pts=256,
            init_dim=3,
            use_dir_cond=False,
            num_fp16_res=4,
            options={},
        ):
        super().__init__()
        
        # TODO implement channel_base
        channel_multiplier = 2
        channels = {
            4: 512,                         # 256
            8: 512,                         # 1024
            16: 256 * channel_multiplier,   # 4096
            32: 128 * channel_multiplier,   # 16384
            64: 64 * channel_multiplier,    # 65536
            128: 64 * channel_multiplier,   # 262144
            256: 64 * channel_multiplier,   # 1048576
        }

        use_pe = options['use_start_pe']
        pe_res = np.square(512)
        if use_pe and init_dim == 3:
            self.conv_in = CoordInjection_const(pe_dim=channels[4] // 2, pe_res=pe_res)
        else:
            self.conv_in = FullyConnectedLayer(init_dim, channels[4])
             
        self.blocks = nn.ModuleDict()
        self.out_norm = nn.ModuleDict()
        self.out_layers = nn.ModuleDict()
        
        weight_init = 0.1
        linear_op = FullyConnectedLayer
        
        # {key: [out_dim, weight_init, lr_mult]}
        _out_keys = {"xyz": [3, 1.0, 1.0], 
                     "scale": [3, weight_init, 1.], 
                     "rotation": [4, weight_init, 1.], 
                     "color": [img_channels, 1, 1.], 
                     "opacity": [1, 1, 1.]}
        
        self._out_keys = _out_keys
        for k in _out_keys.keys():
            self.out_layers[k] = nn.ModuleDict()
        self.gammas = nn.ParameterDict()
        
        if use_dir_cond:
            camera_cond_dim = 32
            self.encoder_camera_cond = FullyConnectedLayer(3, camera_cond_dim, activation='lrelu')
        else:
            camera_cond_dim = 0
        
        # For layer-wise latent code injection
        res_start = 4
        num_pts_start = init_pts
        self.upsample_res = res_start
        img_resolution = options['res_end']
        self.upsample_ratio = options['up_ratio']
        
        self.num_ws = 0
        self.res_fp16_start = 2 ** (np.log2(img_resolution) - num_fp16_res)
        
        self.block_res = []
        while res_start < img_resolution:
            i = res_start
            cur_in, cur_out = channels[i], channels[i*2]
            self.blocks[f"block_{i}"] = nn.ModuleList([])
            self.out_norm[f"block_{i}"] = Identity()
            pe_dim = 0
            if i > self.upsample_res:
                # Attention - Upsample - MLP - Output
                if num_pts_start == 4096 or num_pts_start == 2048:
                    self.ws_stop_res = res_start
                
                # Attention layer
                if num_pts_start <= 4096:
                    self.blocks[f"block_{i}"].append(AdaEdgeBlock(cur_in, cur_in, k=knn)) # TODO
                    self.num_ws += 1
                else:
                    self.blocks[f"block_{i}"].append(LocalEdgeBlock(cur_in, cur_in))
                    self.num_ws += 0
                    
                # Upsample layer
                self.blocks[f"block_{i}"].append(PointUpsample_subpixel(in_features=cur_in, out_features=cur_out, pe_dim=pe_dim, \
                                            upsample_ratio=self.upsample_ratio, resolution=pe_res))

                if num_pts_start <= 4096: # leq level 3
                    self.blocks[f"block_{i}"].append(AdaMLP(width=cur_out))
                    self.num_ws += 1
                else:
                    self.blocks[f"block_{i}"].append(PlainMLP(width=cur_out))
                    self.num_ws += 0
                    
            else:
                # Transformer layer
                self.n_transformer = 6
                self.blocks[f"block_{i}"].append(Transformer(width=cur_in, layers=self.n_transformer))
                
                self.num_ws_transformer = self.n_transformer * 2
                self.num_ws += self.num_ws_transformer # same latent code for transformer / pre and post norm
                
        
            self.output_multiplier = options['output_multiplier']
            for k in _out_keys.keys():
                self.out_layers[k][f"block_{i}"] = nn.ModuleList([])
                
                # Anchor gaussian layer
                self.out_layers[k][f"block_{i}"].append(linear_op(cur_out, _out_keys[k][0], lr_multiplier=_out_keys[k][2], activation="linear", weight_init=_out_keys[k][1]))
                
                # Output gaussian layer
                if k == 'color':
                    cur_in_channel = cur_out + camera_cond_dim
                else:
                    cur_in_channel = cur_out
                self.out_layers[k][f"block_{i}"].append(linear_op(cur_in_channel, _out_keys[k][0] * self.output_multiplier, lr_multiplier=_out_keys[k][2], activation="linear", weight_init=_out_keys[k][1]))
                
                self.num_ws += 0 # no latent code for output layer
            
            self.block_res.append(i)
            res_start = res_start * 2
            num_pts_start = num_pts_start * self.upsample_ratio

        self.register_buffer("scale_init", torch.ones([3]) * options['scale_init'])
        self.register_buffer("scale_threshold", torch.ones([3]) * options['scale_threshold'])
        self.register_buffer("rotation_init", torch.tensor([1, 0, 0, 0]))
        self.register_buffer("color_init", torch.tensor(torch.zeros([img_channels])))
        self.register_buffer("opacity_init", inverse_sigmoid(0.1 * torch.ones([1])))
        
        self.scale_upsample_reg = options['scale_upsample_reg'] # default: 0.0
        self.xyz_output_scale = options['xyz_output_scale'] # default: 0.1
        self.color_output_scale = 1.0
        
        
        
        # TODO check threshold for split
        min_scale = np.exp(options['scale_end'])
        self.percent_dense = min_scale # if scale is less than min_scale, do not apply split
        num_upsample = np.log2(img_resolution) - np.log2(self.upsample_res)
        self.split_ratio = np.exp((options['scale_init'] - np.log(min_scale)) / (num_upsample - 1))
        
        self.scale_out_delta = 0
        self.use_dir_cond = use_dir_cond
        
        self.camera_dep_affine = FullyConnectedLayer(camera_cond_dim, channels[4], bias_init=1.)
        
    def forward(self, x, ws, camera_cond=None, camera_swap_p=0.0):
        # x: [B, L, C]
        B, L, C = x.shape
        pos_sphere = x
        # w = ws[:, 0] # usage for debugging
        
        if len(ws.shape) == 2:
            ws = ws.unsqueeae(1).repeat(1, self.num_ws, 1) # ws: [B, num_ws, w_dim]
        
        ws_transformer = ws[:, :self.num_ws_transformer] # make iter in transformer
        ws = iter(ws[:, self.num_ws_transformer:].unbind(1))
    
        if self.use_dir_cond:
            # TODO swap camera condition
            if camera_swap_p > 0:
                camera_cond_swapped = torch.roll(camera_cond.clone(), 1, 0)
                camera_cond = torch.where(torch.rand((camera_cond.shape[0], 1), device=camera_cond.device) < camera_swap_p, camera_cond_swapped, camera_cond)
            camera_cond = self.encoder_camera_cond(camera_cond)
        
        xyz_prev, scale_prev, rotation_prev, color_prev, opacity_prev = 0, self.scale_init, self.rotation_init, self.color_init, self.opacity_init
        
        upsample_factor = 1
        
        for i in self.block_res:
            if i > self.upsample_res:
                
                if i <= self.ws_stop_res:
                    w_attn, w_up, w_mlp = next(ws), None, next(ws) # previous output
                else:
                    w_attn, w_up, w_mlp = None, None, None
                
                layer_attn, upsample, layer_mlp = self.blocks[f"block_{i}"]
                
                # TODO FP32 to FP16
                if i >= self.res_fp16_start:
                    x = x.to(torch.float16)

                x = layer_attn(x, w=w_attn, pos=xyz_prev, upsample_factor=upsample_factor)
                    
                x, (xyz_prev, scale_prev, rotation_prev, color_prev, opacity_prev) = upsample(x, [xyz_prev, scale_prev, rotation_prev, color_prev, opacity_prev], w=w_up)
                upsample_factor *= self.upsample_ratio
                
                x = layer_mlp(x, w=w_mlp)
                
            else:
                # Start resolution
                layer_transformer = self.blocks[f"block_{i}"][0]
                x = self.conv_in(x)
           
                x = layer_transformer(x, ws_transformer)
            
            out = x # out for output layers, x for next level layers
             
            # TODO FP16 to FP32 (for output layers)
            if i >= self.res_fp16_start:
                out = out.to(torch.float32)
            
            # Last layer (do not generator anchor gaussians)
            if i == self.block_res[-1]:
                
                # Point generation part
                pc = [self.out_layers[k][f"block_{i}"][1](out) for k in ["xyz", "scale", "rotation", "opacity"]]
                
                if self.use_dir_cond:
                    out_color = torch.cat([out, camera_cond.unsqueeze(1).repeat(1, out.shape[1], 1)], dim=-1)
                else:
                    out_color = out
                pc.insert(3, self.out_layers['color'][f"block_{i}"][1](out_color))
                
                pc_prev = [xyz_prev, scale_prev, rotation_prev, color_prev, opacity_prev]
                
                # Postprocessing (densify)
                xyz_cur, scale_cur, rotation_cur, color_cur, opacity_cur = self.postprocessing_block(pc, pc_prev, percent_dense=self.percent_dense, \
                                                                            N=self.upsample_ratio, split_ratio=self.split_ratio, \
                                                                            scale_upsample_reg=self.scale_out_delta)
                
                xyz = torch.cat([xyz, xyz_cur], dim=1)
                scale = torch.cat([scale, scale_cur], dim=1)
                rotation = torch.cat([rotation, rotation_cur], dim=1)
                color = torch.cat([color, color_cur], dim=1)
                opacity = torch.cat([opacity,  opacity_cur], dim=1)

            # Intermediate layers (generate both anchor and output gaussians)
            elif i > self.upsample_res:
                pc = [self.out_layers[k][f"block_{i}"][0](out) for k in ["xyz", "scale", "rotation", "color", "opacity"]]
                pc_prev = [xyz_prev, scale_prev, rotation_prev, color_prev, opacity_prev]
                
                xyz_cur, scale_cur, rotation_cur, color_cur, opacity_cur = self.postprocessing_block(pc, pc_prev, percent_dense=self.percent_dense, \
                                                                            N=self.upsample_ratio, split_ratio=self.split_ratio)

                # Point generation part
                pc_cur_out = [self.out_layers[k][f"block_{i}"][1](out) for k in ["xyz", "scale", "rotation", "opacity"]]
                
                if self.use_dir_cond:
                    out_color = torch.cat([out, camera_cond.unsqueeze(1).repeat(1, out.shape[1], 1)], dim=-1)
                else:
                    out_color = out
                    
                pc_cur_out.insert(3, self.out_layers['color'][f"block_{i}"][1](out_color))
                
                if self.output_multiplier > 1:
                    pc_new, pc_prev_new = [], []
                    for _comp, _comp_prev in zip(pc_cur_out, pc_prev):
                        _b, _l, _c = _comp_prev.shape
                        pc_new.append(_comp.reshape(_b, _l * self.output_multiplier, _c))
                        pc_prev_new.append(torch.repeat_interleave(_comp_prev, self.output_multiplier, dim=1))
                    pc_cur_out, pc_prev = pc_new, pc_prev_new
                
                # Postprocessing (densify)
                xyz_cur_out, scale_cur_out, rotation_cur_out, color_cur_out, opacity_cur_out = self.postprocessing_block(pc_cur_out, pc_prev, percent_dense=self.percent_dense, \
                                                                            N=self.upsample_ratio, split_ratio=self.split_ratio, \
                                                                            scale_upsample_reg=self.scale_out_delta)
                
                xyz = torch.cat([xyz, xyz_cur_out], dim=1)
                scale = torch.cat([scale, scale_cur_out], dim=1)
                rotation = torch.cat([rotation, rotation_cur_out], dim=1)
                color = torch.cat([color, color_cur_out], dim=1)
                opacity = torch.cat([opacity, opacity_cur_out], dim=1)
                                
                xyz_anchor = torch.cat([xyz_anchor, xyz_cur], dim=1)
                scale_anchor = torch.cat([scale_anchor, scale_cur], dim=1)
                rotation_anchor = torch.cat([rotation_anchor, rotation_cur], dim=1)
                color_anchor = torch.cat([color_anchor, color_cur], dim=1)
                opacity_anchor = torch.cat([opacity_anchor, opacity_cur], dim=1)
            
            # First layer (does not have previous anchors and upsamples, so generate both of them)
            else:
                xyz_cur = xyz_prev + self.out_layers["xyz"][f"block_{i}"][0](out) * self.xyz_output_scale # TODO small output
                scale_cur = scale_prev + self.out_layers["scale"][f"block_{i}"][0](out)
                rotation_cur = rotation_prev + self.out_layers["rotation"][f"block_{i}"][0](out)
                color_cur = color_prev + self.out_layers["color"][f"block_{i}"][0](out) * self.color_output_scale
                opacity_cur = opacity_prev + self.out_layers["opacity"][f"block_{i}"][0](out)

                # Point generation part
                xyz_cur_out = self.out_layers["xyz"][f"block_{i}"][1](out).reshape(B, L * self.output_multiplier, -1)
                scale_cur_out = scale_prev + self.out_layers["scale"][f"block_{i}"][1](out).reshape(B, L * self.output_multiplier, -1)
                rotation_cur_out = rotation_prev + self.out_layers["rotation"][f"block_{i}"][1](out).reshape(B, L * self.output_multiplier, -1)
                out_color = torch.cat([out, camera_cond.unsqueeze(1).repeat(1, out.shape[1], 1)], dim=-1) if self.use_dir_cond else out
                
                color_cur_out = color_prev + self.out_layers["color"][f"block_{i}"][1](out_color).reshape(B, L * self.output_multiplier, -1) * self.color_output_scale
                opacity_cur_out = opacity_prev + self.out_layers["opacity"][f"block_{i}"][1](out).reshape(B, L * self.output_multiplier, -1)
                
                
                # Postprocessing (densify)
                if i == self.upsample_res:
                    xyz_cur = torch.tanh(xyz_cur)
                    scale_cur = - torch.nn.functional.softplus(- (scale_cur - self.scale_threshold)) + self.scale_threshold

                    if self.output_multiplier > 1:
                        rotation_cur_temp = torch.repeat_interleave(rotation_cur, self.output_multiplier, dim=1)
                        scale_cur_temp = torch.repeat_interleave(scale_cur, self.output_multiplier, dim=1)
                        xyz_cur_temp = torch.repeat_interleave(xyz_cur, self.output_multiplier, dim=1)
                    else:
                        rotation_cur_temp = rotation_cur
                        scale_cur_temp = scale_cur
                        xyz_cur_temp = xyz_cur

                    samples = torch.tanh(xyz_cur_out) # [B, L * N, 3]
                    R = get_scaled_directional_vector_from_quaternion(rotation_cur_temp, scale_cur_temp) # [B, L * N, 3, 3]
                    
                    xyz_cur_out = (R @ samples.unsqueeze(-1)).squeeze(-1) + xyz_cur_temp # [B, L * N, 3]
                    scale_cur_out = - torch.nn.functional.softplus(- (scale_cur_out - self.scale_threshold - self.scale_out_delta)) + self.scale_threshold + self.scale_out_delta
                    
                    
                # TODO check xyz, ... is output of point generator
                xyz_anchor, scale_anchor, rotation_anchor, color_anchor, opacity_anchor = xyz_cur, scale_cur, rotation_cur, color_cur, opacity_cur
                xyz, scale, rotation, color, opacity = xyz_cur_out, scale_cur_out, rotation_cur_out, color_cur_out, opacity_cur_out
                
            # TODO check xyz_prev, ... is point clouds using for next layer
            # update previous gaussian params
            xyz_prev, scale_prev, rotation_prev, color_prev, opacity_prev = xyz_cur, scale_cur, rotation_cur, color_cur, opacity_cur
                
        # Output phase
        B, L, _ = xyz.shape
        xyz, scale, rotation, color, opacity = xyz.view(B, L, -1), scale.view(B, L, -1), \
                                            rotation.view(B, L, -1), color.view(B, L, -1), opacity.view(B, L, -1)
        xyz = torch.clamp(xyz, -1, 1)
        
        return xyz, scale, rotation, color, opacity, [xyz_anchor, scale_anchor, rotation_anchor, color_anchor, opacity_anchor]
    

    def postprocessing_block(self, pc, pc_prev, percent_dense=0.05, N=2, split_ratio=None, scale_upsample_reg=0):
        xyz, scale, rotation, color, opacity = pc_prev # previous outputs
        xyz_cur, scale_cur, rotation_cur, color_cur, opacity_cur = pc # current outputs
    
        rotation_new = rotation + rotation_cur
        color_new = color + color_cur * self.color_output_scale
        opacity_new = opacity + opacity_cur

        samples = torch.tanh(xyz_cur) # [B, L * N, 3]
        R = get_scaled_directional_vector_from_quaternion(rotation, scale) # [B, L * N, 3, 3]
        
        xyz_new = (R @ samples.unsqueeze(-1)).squeeze(-1) + xyz # [B, L * N, 3]
        
        # split and clone w/ prev scale
        if split_ratio is None:
            scale_split = torch.log(torch.exp(scale) / (0.8 * N)) # default setting of 3DGS
        else:
            scale_split = torch.log(torch.exp(scale) / split_ratio)
        
        # remove -inf
        scale_split = torch.clamp(scale_split, -1e+2, 0)
        
        scale_max = torch.exp(scale).max(-1, keepdim=True)[0]
        split_idx = torch.where(scale_max > percent_dense, 1., 0.)
        
        scale = scale * (1. - split_idx) + scale_split * split_idx
        scale_new = scale + -1 * torch.nn.functional.softplus(- (scale_cur - scale_upsample_reg)) + scale_upsample_reg
        
        # remove outliers
        scale_new = torch.clamp(scale_new, -8, 0)
        
        pc_upsampled = [xyz_new, scale_new, rotation_new, color_new, opacity_new]
        return pc_upsampled
    
    
    def mutiply_quaternion(self, q1, q2):
        q1, q2 = torch.nn.functional.normalize(q1, dim=-1), torch.nn.functional.normalize(q2, dim=-1)
        
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        
        w_product = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x_product = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y_product = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z_product = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        
        q_new = torch.stack([w_product, x_product, y_product, z_product], dim=-1)
        return torch.nn.functional.normalize(q_new, dim=-1)
        