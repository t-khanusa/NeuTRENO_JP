"""
Adapted from 2020 Ross Wightman
https://github.com/rwightman/pytorch-image-models
"""

import torch
import torch.nn as nn
from einops import rearrange
from pathlib import Path

import torch.nn.functional as F

from timm.models.layers import DropPath


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout, out_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        if out_dim is None:
            out_dim = dim
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop = nn.Dropout(dropout)

    @property
    def unwrapped(self):
        return self

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads, dropout, layerth = None):
        super().__init__()
        self.heads = heads
        head_dim = dim // heads
        self.scale = head_dim ** -0.5
        self.attn = None

        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        self.layerth = layerth
        self.alpha = 0.5
        self.alpha_1 = 0.9
        self.alpha_2 = 0.3
        self.alpha_3 = 0.0
        self.alpha_all = 1.
        print(self.alpha, self.alpha_2, self.alpha_3)

    @property
    def unwrapped(self):
        return self

    def forward(self, x, mask=None, v0 = None, accum_res_prev = None, u_k2 = None, u_k1 = None):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.heads, C // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)



        diff = (u_k2 - u_k1) if self.layerth > 1 else 0.
        if self.layerth == 0:
            res = 0.
        else:
            res = self.alpha*(v0 - v) + self.alpha_2*(accum_res_prev) + self.alpha_3*diff
        x = (attn @ v) + self.alpha_all*(res) 
        accum_res_prev = self.alpha_1*(v0 - x) + (1 - self.alpha_1)*(accum_res_prev) if self.layerth > 0 else v - x     
        u_k2  = u_k1
        u_k1 = x

        x = x.transpose(1, 2).reshape(B,N,C) 
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.layerth == 0:
            return x, attn, v, accum_res_prev, u_k1, u_k2
        else:
            return x, attn, accum_res_prev, u_k1, u_k2


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, drop_path, layerth = None):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads, dropout, layerth= layerth)
        self.layerth = layerth
        self.mlp = FeedForward(dim, mlp_dim, dropout)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, mask=None, return_attention=False, v0 = None, accum_res_prev = None, u_k2 = None, u_k1 = None):
        if self.layerth == 0:
            y, attn, v0, accum_res_prev, u_k1, u_k2 = self.attn(self.norm1(x), mask)
        else:
            y, attn, accum_res_prev, u_k1, u_k2= self.attn(self.norm1(x), mask, v0 = v0, accum_res_prev = accum_res_prev, u_k1 = u_k1, u_k2 = u_k2)
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.layerth == 0:
            return x, v0, accum_res_prev, u_k1, u_k2
        else:
            return x, accum_res_prev, u_k1, u_k2
