import torch
import torch.nn as nn
from torch_utils import persistence
from torch_utils import misc

import math
import numpy as np

from training.networks_stylegan2 import FullyConnectedLayer
from training.checkpoint import checkpoint

def init_linear(l, stddev):
    nn.init.normal_(l.weight, std=stddev)
    if l.bias is not None:
        nn.init.constant_(l.bias, 0.0)

@persistence.persistent_class
class InstanceNorm1d(nn.Module):
    def __init__(
        self,
        dim,
        affine=False,
    ):
        super().__init__()
        # Instance Normalization with [N, L, C] format and no affine transformation
        pass
        
    def forward(self, x):
        # Input as [B, L, C]
        dtype = x.dtype
        x = x.to(torch.float32)
        x = (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True, unbiased=False) + 1e-8)
        return x.to(dtype)
        
    
@persistence.persistent_class
class AdaptiveNorm(nn.Module):
    def __init__(
        self,
        dim,
        normtype='instance',
        weight_init=1.0,
    ):
        super().__init__()
        
        assert normtype in ['layer', 'instance', 'none']
        
        w_dim = 512 # global latent
        
        self.gamma = FullyConnectedLayer(w_dim, dim, activation='linear', weight_init=weight_init, bias_init=1.)
        self.beta = FullyConnectedLayer(w_dim, dim, activation='linear', weight_init=weight_init, bias_init=0.)
        
        if normtype == 'layer':
            self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        elif normtype == 'instance':
            self.norm = InstanceNorm1d(dim, affine=False)
        elif normtype == 'none':
            self.norm = nn.Identity()

    def forward(self, x, w):
        if len(w.shape) == 2:
            # Global latent
            return self.norm(x) * self.gamma(w).unsqueeze(1) + self.beta(w).unsqueeze(1)
        else:
            # Local latent
            return self.norm(x) * self.gamma(w) + self.beta(w)

@persistence.persistent_class
class MultiheadAttention(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.heads = heads
        
        self.attention = QKVMultiheadAttention(heads=heads, n_ctx=n_ctx)
        self.c_qkv = FullyConnectedLayer(width, width * 3)
        
        self.c_proj = FullyConnectedLayer(width, width)

    def forward(self, x):
        x = self.c_qkv(x)
        x = checkpoint(self.attention, (x,), (), True)
        x = self.c_proj(x)
        return x


@persistence.persistent_class
class MLP(nn.Module):
    def __init__(self, *, width: int, init_scale: float):
        super().__init__()
        self.width = width
        self.gelu = nn.GELU()

        self.c_fc = FullyConnectedLayer(width, width * 4)
        self.c_proj = FullyConnectedLayer(width * 4, width)

    def forward(self, x, w=None):
        return self.c_proj(self.gelu(self.c_fc(x)) * np.sqrt(2))

@misc.profiled_function
def pixel_norm(x: torch.FloatTensor, eps=1e-4, dim=1):
    return x / torch.sqrt(torch.mean(x ** 2, dim=dim, keepdim=True) + eps)

@persistence.persistent_class
class QKVMultiheadAttention(nn.Module):
    def __init__(self, *, heads: int, n_ctx: int):
        super().__init__()
        self.heads = heads
        self.n_ctx = n_ctx

    def forward(self, qkv):
        bs, n_ctx, width = qkv.shape
        attn_ch = width // self.heads // 3
        scale = 1 / math.sqrt(math.sqrt(attn_ch))
        qkv = qkv.view(bs, n_ctx, self.heads, -1)
        q, k, v = torch.split(qkv, attn_ch, dim=-1)
        
        # Dot product attention
        weight = torch.einsum(
            "bthc,bshc->bhts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        wdtype = weight.dtype
        weight = torch.softmax(weight.float(), dim=-1).type(wdtype)
        return torch.einsum("bhts,bshc->bthc", weight, v).reshape(bs, n_ctx, -1)


@persistence.persistent_class
class ResidualAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int,
        width: int,
        heads: int,
        init_scale: float = 1.0,
        use_postnorm: bool = False,
    ):
        super().__init__()

        self.use_postnorm = use_postnorm

        self.attn = MultiheadAttention(
            n_ctx=n_ctx,
            width=width,
            heads=heads,
            init_scale=init_scale,
        )
        self.mlp = MLP(width=width, init_scale=init_scale)
        
        self.ln_1 = AdaptiveNorm(width)
        self.ln_2 = AdaptiveNorm(width)
        
        self.ls_1 = FullyConnectedLayer(512, width, activation='linear', weight_init=0.)
        self.ls_2 = FullyConnectedLayer(512, width, activation='linear', weight_init=0.)

    def forward(self, x: torch.Tensor, w: torch.Tensor):

        x = x + self.attn(self.ln_1(x, w[0])) * self.ls_1(w[0]).unsqueeze(1)
        x = x + self.mlp(self.ln_2(x, w[1])) * self.ls_2(w[1]).unsqueeze(1)
        
        return x
    

@persistence.persistent_class
class Transformer(nn.Module):
    def __init__(
        self,
        *,
        n_ctx: int = 256,
        width: int = 512,
        layers: int = 2,
        heads: int = 8,
        init_scale: float = 0.25,
    ):
        super().__init__()
        self.n_ctx = n_ctx
        self.width = width
        self.layers = layers
        init_scale = init_scale * math.sqrt(1.0 / width)
        self.resblocks = nn.ModuleList(
            [
                ResidualAttentionBlock(
                    n_ctx=n_ctx,
                    width=width,
                    heads=heads,
                    init_scale=init_scale,
                )
                for _ in range(layers)
            ]
        )
        self.num_ws = self.layers * 2
    
    def forward(self, x: torch.Tensor, ws: torch.Tensor):
        
        # Global latent
        if len(ws.shape) == 2:
            ws = ws.unsqueeze(1).repeat(1, self.num_ws, 1)
        
        ws = iter(ws.unbind(1))
        
        for block in self.resblocks:
            w_attn, w_mlp = next(ws), next(ws)
            x = block(x, [w_attn, w_mlp])
        
        return x
