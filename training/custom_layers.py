import torch
import torch.nn as nn
import numpy as np
import math
from itertools import islice

from training.networks_stylegan2 import FullyConnectedLayer, Conv2dLayer, SynthesisLayer

from torch_utils import misc
from torch_utils import persistence
from torch_utils.ops import bias_act
import torch.nn.init as init

import torch.nn.functional as F


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
class StyleInjector(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        w_dim,                      # Number of w features
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        clamp           = 256,
    ):
        super().__init__()
        
        # Only adding modulation (not demodulation)
        
        self.in_features = in_features
        self.activation = activation
        
        bias_init = 0.0
        self.weight = torch.nn.Parameter(torch.eye(in_features, dtype=torch.float32))
        # self.weight = torch.nn.Parameter(torch.ones([in_features, in_features], dtype=torch.float32))
        # self.weight_scale = torch.nn.Parameter(torch.ones([1, in_features], dtype=torch.float32))
        # self.register_buffer("weight_base", torch.eye(in_features, dtype=torch.float32))
        # self.register_buffer("weight", torch.eye(in_features, dtype=torch.float32))
        self.bias = torch.nn.Parameter(torch.full([in_features], np.float32(bias_init))) if bias else None
        self.weight_gain = 1.0
        self.bias_gain = 1.0
        
        self.clamp = clamp
        self.affine = FullyConnectedLayer(w_dim, in_features, weight_init=0.0, bias_init=1)

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
        
        return x

    def extra_repr(self):
        return f'in_features={self.in_features:d}, activation={self.activation:s}'
    

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

    

@persistence.persistent_class
class EdgeBlock(nn.Module):
    # changed from Edgeblock in SPGAN
    def __init__(self, Fin, Fout, k):
        super().__init__()
        self.k = k
        self.Fin = Fin
        self.Fout = Fout
        
        self.conv_w = nn.Sequential(
            Conv2dLayer(Fin, Fout//2, kernel_size=1, activation='lrelu'),
            Conv2dLayer(Fout//2, Fout, kernel_size=1, activation='linear'),
        )

        # self.conv_x = Conv2dLayer(2 * Fin, Fout, kernel_size=1, activation='lrelu')
        self.conv_x = nn.Sequential(
            Conv2dLayer(2 * Fin, Fout, kernel_size=1, activation='lrelu'),
            Conv2dLayer(Fout, Fout, kernel_size=1, activation='linear'),
        )

    def forward(self, x, w=None, pos=None, upsample_factor=None):
        # x: [B, L, C]
        x_original = x
        
        x = x.permute(0, 2, 1) # [B, C, L]
        
        B, C, N = x.shape
        x = get_edge_features(x, self.k) # [B, 2Fin, N, k]
        
        w = self.conv_w(x[:, C:, :, :])
        
        w = F.softmax(w.to(torch.float32), dim=-1)  # [B, Fout, N, k] -> [B, Fout, N, k]
        w = w.to(x.dtype)

        x = self.conv_x(x)  # Bx2CxNxk
        
        x = (x * w).sum(-1, keepdim=True)

        x = x.squeeze(3)  # BxCxN
        
        return x.permute(0, 2, 1)