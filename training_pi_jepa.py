"""
PI(D)-JEPA: collapse prevention by feedback control at BOTH the token level AND
the sample level — no SIGReg, no teacher/EMA. The ONLY training loss is multi-view
invariance plus the control force.

  1. Token-level PID (PIDformer backbone, Nguyen et al. 2024)  [token_pid flag]
     Controls the token field u(x,t) along DEPTH toward the input reference.
     Prevents token/rank collapse (oversmoothing) WITHIN a sample. When
     token_pid=false, a standard timm ViT backbone is used instead.

  2. Sample-level channel-split PID (CovariancePID)
     Controls the batch covariance C of the L2-normalized projection along
     TRAINING time. The control law enters as a gradient (loss term) — the only
     channel that reaches the plant (a post-hoc actuator cannot prevent collapse:
     total collapse is a zero-loss optimum of invariance, so the force must be in
     the objective). The covariance is split into two channels with DIFFERENT
     control objectives.

═══════════════════════════════════════════════════════════════════════════════
SAMPLE-LEVEL CONTROL — channel split (the contribution)
═══════════════════════════════════════════════════════════════════════════════

Plant      : encoder f_θ ; output y = C = Cov_s[ẑ]  (ẑ L2-normalized → bounded).
Loss form  : non-negative quadratic tracking, ½·k_p·‖C − R_eff‖² per channel,
             with the I/D action realized as an effective setpoint shift R_eff.

DIAGONAL (variance) — INEQUALITY constraint C_ii ≥ σ²/dim
    Collapse is one-sided (variance too LOW); above the floor is success.
    → PROPORTIONAL DEADBAND, NO integral.  Integrating a rectified one-sided
      error integrates finite-batch noise (E[relu(R−C_ii)]>0 even at setpoint)
      → unbounded windup (this was the observed total_loss ramp). A small
      steady-state offset below the floor is acceptable (we need a bound, not
      exact tracking).
        L_var = ½·k_p · Σ_i relu(σ²/dim − C_ii)²

OFF-DIAGONAL (correlation) — EQUALITY setpoint C_ij = 0
    Exactness matters (redundancy reduction); error is zero-mean → full PID
    is well-posed:
        e = −C_off
        W   ← clamp(W + η·e, ±w_max)            (Integral state)
        dE  = filt(e) − filt(e)_prev            (Derivative, EMA-filtered)
        R_eff = (k_i·W + k_d·dE)/k_p            (PID setpoint shift, detached)
        L_cov = ½·k_p · Σ_{i≠j}(C_ij − R_eff_ij)²
    ⇒ ∂L_cov/∂C_off = k_p·C_off − k_i·W − k_d·dE = −(PID control signal).
    • I  → drives the time-averaged correlation to EXACTLY zero (no residual bias).
    • D  → damping; on a filtered measurement to avoid per-batch noise blow-up
           (Aström §6, "derivative on filtered measurement"). cov_d_filter<1 smooths.

Why this beats prior collapse-prevention regularizers (all are memoryless = P):
    VICReg / VCReg : P  (variance hinge + Σ C_ij²)        → residual corr. bias
    Barlow Twins   : P  (cross-correlation → I)           → residual bias
    SIGReg (LeJEPA): P  (match sliced N(0,I) distribution) → residual bias
    This (PID)     : P deadband (variance) + PID (decorr.) → EXACT decorrelation +
                     disturbance rejection (steady state independent of the
                     invariance-loss weighting → self-tuning).

Stability (linearized off-diagonal loop, per element):
    (1+k_d) s² + (g + k_p) s + k_i = 0  → Routh–Hurwitz stable for positive gains.
═══════════════════════════════════════════════════════════════════════════════

Usage:
    python training_pi_jepa.py                      # uses configs/pi_jepa.yaml
    python training_pi_jepa.py token_pid=true       # PIDformer backbone
    python training_pi_jepa.py cov_use_integral=false cov_use_derivative=false  # P-only ablation
"""

import os
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP

import timm
import timm.layers
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_
import wandb
import hydra
import tqdm
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# TOKEN-LEVEL PID — PIDformer backbone (exact author implementation)
# Same parameter names as standard ViT → direct pretrained weight loading.
# PID gains are fixed constants, NOT learned parameters.
# ─────────────────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 layerth=None, alpha=None):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # PID gains (paper Appendix A.1)
        self.alpha = 0.8        # λ_P (Proportional)
        self.alpha_1 = 0.3      # Integral EMA factor
        self.alpha_2 = 0.5      # λ_I (Integral)
        self.alpha_3 = 0.05     # λ_D (Derivative)
        self.alpha_v0 = 0.1     # β (reference scaling)
        self.layerth = layerth

    def forward(self, x, v0=None, accum_res_prev=None, u_k2=None, u_k1=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        diff = (u_k2 - u_k1) if self.layerth > 1 else 0.

        if self.layerth == 0:
            res = 0.
        else:
            v0 = self.alpha_v0 * v0
            res = self.alpha * (v0 - v) + self.alpha_2 * (accum_res_prev) + self.alpha_3 * diff

        x = (attn @ v) + (res)
        accum_res_prev = self.alpha_1 * (v0 - v) + (1 - self.alpha_1) * (accum_res_prev) if self.layerth > 0 else 0.

        u_k2 = u_k1
        u_k1 = v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if self.layerth == 0:
            return x, v, accum_res_prev, u_k1, u_k2
        else:
            return x, accum_res_prev, u_k1, u_k2


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, layerth=None, alpha=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              attn_drop=attn_drop, proj_drop=drop, layerth=layerth, alpha=alpha)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.layerth = layerth

    def forward(self, x, v0=None, accum_res_prev=None, u_k1=None, u_k2=None):
        if self.layerth == 0:
            x_, v0, accum_res_prev, u_k1, u_k2 = self.attn(self.norm1(x))
        else:
            x_, accum_res_prev, u_k1, u_k2 = self.attn(
                self.norm1(x), v0=v0, accum_res_prev=accum_res_prev, u_k1=u_k1, u_k2=u_k2)

        x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        if self.layerth == 0:
            return x, v0, accum_res_prev, u_k1, u_k2
        else:
            return x, accum_res_prev, u_k1, u_k2


class PIDformer(nn.Module):
    """PIDformer ViT — same structure as timm's VisionTransformer."""

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, act_layer=None, alpha=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                  drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer, layerth=i, alpha=alpha)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.pre_logits = nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        x, v0, accum_res_prev, u_k1, u_k2 = self.blocks[0](x)
        for i in range(1, len(self.blocks)):
            x, accum_res_prev, u_k1, u_k2 = self.blocks[i](
                x, v0=v0, accum_res_prev=accum_res_prev, u_k1=u_k1, u_k2=u_k2)

        x = self.norm(x)
        return self.pre_logits(x[:, 0])

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def load_pretrained_from_timm(self, model_name="vit_small_patch8_224", img_size=128):
        print(f"Loading pretrained weights from timm '{model_name}'...")
        ref = timm.create_model(model_name, pretrained=True, img_size=img_size)
        ref_sd = ref.state_dict()
        own_sd = self.state_dict()
        loaded = {k: v for k, v in ref_sd.items() if k in own_sd and own_sd[k].shape == v.shape}
        missing = set(own_sd.keys()) - set(loaded.keys())
        self.load_state_dict(loaded, strict=False)
        print(f"  Loaded {len(loaded)}/{len(own_sd)} parameters. Skipped: {sorted(missing)}")
        del ref


# ─────────────────────────────────────────────────────────────────────────────
# Standard ViT Encoder (timm) — used when token_pid=false
# ─────────────────────────────────────────────────────────────────────────────

class ViTEncoder(nn.Module):
    """Standard timm encoder (ViT, ResNet, etc.) with an MLP projector head."""

    def __init__(self, model_name="vit_small_patch8_224", img_size=128,
                 embed_dim=512, proj_dim=16, drop_path_rate=0.1, pretrained=False):
        super().__init__()
        kwargs = dict(
            pretrained=pretrained,
            num_classes=embed_dim,
            drop_path_rate=drop_path_rate,
        )
        # CNN backbones (e.g. resnet18d) do not accept img_size.
        if "resnet" not in model_name.lower():
            kwargs["img_size"] = img_size
        self.backbone = timm.create_model(model_name, **kwargs)
        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE-LEVEL FEEDBACK CONTROL — Channel-split PID covariance controller
#
# Plant : the encoder f_θ ; plant output y(τ) = C(τ) = Cov_s[ẑ_s]  (batch cov of
#         the L2-normalized projection → scale-invariant, cannot blow up).
# The control law enters as a gradient (a loss term) — the only channel that
# reaches the plant. We SPLIT the covariance into two channels because they carry
# DIFFERENT control objectives:
#
#   DIAGONAL  (variance)   →  an INEQUALITY constraint  C_ii ≥ σ²/dim
#       Collapse is a one-sided failure (variance too LOW); being above the floor
#       is success, not error. So this channel uses a PROPORTIONAL DEADBAND (hinge)
#       and NO integral. Integrating a rectified (one-sided) error integrates
#       finite-batch noise — E[relu(R−C_ii)] > 0 even at the setpoint — so the
#       integral would wind up without bound (observed empirically). A small
#       steady-state offset below the floor is acceptable (we need a lower bound,
#       not exact tracking).
#
#   OFF-DIAGONAL (correlation) →  an EQUALITY setpoint  C_ij = 0
#       Exactness matters (redundancy reduction). The error is symmetric / zero-
#       mean, so the full PID is well-posed here:
#         • Integral (I): drives the TIME-AVERAGED correlation to EXACTLY zero,
#           removing the residual steady-state bias that P-only penalties leave.
#         • Derivative (D): damping term; computed on an (optionally EMA-filtered)
#           error to mitigate the per-batch measurement-noise amplification that
#           raw discrete differentiation suffers (Aström §6, "derivative on
#           filtered measurement"). Set d_filter<1 to smooth, =1 for raw dE.
#
# Realization as a non-negative loss (PID = effective setpoint shift):
#   L_var = ½·k_p · Σ_i  relu(σ²/dim − C_ii)²                         (P deadband)
#   L_cov = ½·k_p · Σ_{i≠j} (C_ij − R_eff_ij)²                        (P + I + D)
#   R_eff = ( k_i·W  +  k_d·dE ) / k_p ,   W ← W + η·(−C_off)          (off-diag)
#   ⇒ ∂L_cov/∂C_off = k_p·C_off − k_i·W − k_d·dE = −(PID control signal).
#
# Difference from prior collapse-prevention regularizers — they are all
# memoryless / proportional; this adds TEMPORAL FEEDBACK (an integral state):
#   VICReg / VCReg : P  (variance hinge + Σ C_ij²)        → residual corr. bias
#   Barlow Twins   : P  (cross-correlation → I)           → residual bias
#   SIGReg (LeJEPA): P  (match sliced N(0,I) distribution) → residual bias
#   This (PID)     : P deadband (variance) + PID (decorr.) → EXACT decorrelation,
#                    disturbance rejection (steady state independent of the
#                    invariance-loss weighting → self-tuning).
# ─────────────────────────────────────────────────────────────────────────────

class CovariancePID(nn.Module):
    """Channel-split feedback controller of the batch covariance, applied as a
    control force (loss term). Prevents sample-level collapse with no teacher/EMA
    and no SIGReg.

      • Diagonal (variance)  : proportional DEADBAND — anti-collapse floor only.
      • Off-diagonal (corr.) : P + I + D — exact-zero decorrelation with damping.
    """

    def __init__(self, dim, sigma2=1.0, k_p=1.0, k_i=0.3, k_d=0.05, eta_i=0.02,
                 w_clip=10.0, cov_lambda=1.0, normalize=True,
                 use_integral=True, use_derivative=True, d_filter=1.0, **kwargs):
        super().__init__()
        self.dim = dim
        self.sigma2 = sigma2            # target covariance trace (per-dim floor = sigma2/dim)
        self.k_p = k_p                  # proportional gain (weights both channels)
        self.k_i = k_i                  # integral gain (off-diagonal decorrelation)
        self.k_d = k_d                  # derivative gain (off-diagonal damping)
        self.eta_i = eta_i              # integral accumulation step
        self.w_clip = w_clip            # safety clamp on the integral state
        self.cov_lambda = cov_lambda    # master weight of the control force
        self.normalize = normalize      # L2-normalize features (scale-invariant plant)
        self.use_integral = use_integral      # ablation: enable I action
        self.use_derivative = use_derivative  # ablation: enable D action
        self.d_filter = d_filter        # EMA factor for derivative measurement (1=raw)

        # Off-diagonal channel memory (persistent across training steps).
        self.register_buffer("W_off", torch.zeros(dim, dim))    # integral state
        self.register_buffer("E_off_f", torch.zeros(dim, dim))  # filtered error (for derivative)
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    @staticmethod
    def _sym(A):
        return 0.5 * (A + A.transpose(-1, -2))

    def forward(self, z):
        """z: (M, dim) flattened features.
        Returns (z_normalized, control_loss, diagnostics).
        """
        dtype_in = z.dtype
        z = z.float()
        # L2-normalize onto the unit sphere → the plant is scale-invariant, so the
        # backbone cannot inflate its feature norm (no variance explosion).
        if self.normalize:
            z = F.normalize(z, dim=-1, eps=1e-6)

        if not self.training:
            return z.to(dtype_in), z.new_zeros(()), {}

        M = z.shape[0]
        target_var = self.sigma2 / self.dim   # isotropic per-dim variance floor

        mu = z.mean(dim=0, keepdim=True)
        z_tilde = z - mu
        C = self._sym(z_tilde.transpose(0, 1) @ z_tilde) / max(M - 1, 1)   # plant output (diff'able)

        diagC = torch.diagonal(C)
        off = C - torch.diag_embed(diagC)                                 # off-diagonal part (diff'able)

        # ── Off-diagonal P + I + D: integral + (filtered) derivative memory ──
        with torch.no_grad():
            e_off = -off                                  # error toward setpoint 0
            e_off_f = (1.0 - self.d_filter) * self.E_off_f + self.d_filter * e_off
            dE_off = e_off_f - self.E_off_f if self.steps.item() > 0 else torch.zeros_like(e_off)
            self.E_off_f.copy_(e_off_f)
            if self.use_integral:
                self.W_off.add_(self.eta_i * e_off).clamp_(-self.w_clip, self.w_clip)
            self.steps += 1
            # Effective off-diagonal setpoint shifted by integral + derivative action.
            R_eff_off = (self.k_i * self.W_off
                         + (self.k_d * dE_off if self.use_derivative else 0.0)) / max(self.k_p, 1e-6)

        # ── Non-negative control loss ──
        # Diagonal: one-sided variance floor (P deadband, no integral → no windup).
        var_violation = (target_var - diagC).clamp_min(0.0)
        L_var = 0.5 * self.k_p * var_violation.square().sum()
        # Off-diagonal: P + I + D → drive correlations to exactly zero.
        L_cov = 0.5 * self.k_p * (off - R_eff_off).square().sum()
        ctrl_loss = self.cov_lambda * (L_var + L_cov)

        diag = {
            "ctrl/var_violation": var_violation.mean().item(),
            "ctrl/integral_norm": self.W_off.norm().item(),
            "ctrl/prop_force": (self.k_p * off).detach().norm().item(),
            "ctrl/integral_force": (self.k_i * self.W_off).norm().item(),
            "ctrl/cov_trace": diagC.mean().item(),
            "ctrl/cov_offdiag": off.abs().mean().item(),
            "ctrl/ctrl_loss": ctrl_loss.item(),
        }
        return z.to(dtype_in), ctrl_loss, diag

# ─────────────────────────────────────────────────────────────────────────────
# Encoder: PIDformer backbone + projector
# ─────────────────────────────────────────────────────────────────────────────

class PIDformerEncoder(nn.Module):
    def __init__(self, img_size=128, patch_size=8, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4.0, out_dim=512, proj_dim=16,
                 drop_path_rate=0.1, pretrained=False):
        super().__init__()
        self.backbone = PIDformer(
            img_size=img_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, num_classes=out_dim,
            drop_path_rate=drop_path_rate, qkv_bias=True,
        )
        if pretrained:
            self.backbone.load_pretrained_from_timm(
                model_name="vit_small_patch8_224", img_size=img_size)
        self.proj = MLP(out_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Crop Dataset (identical to the other pipelines)
# ─────────────────────────────────────────────────────────────────────────────

class MultiCropDataset(torch.utils.data.Dataset):
    def __init__(self, split, V_global=2, V_local=4, img_size=128,
                 dataset_name="imagenette"):
        self.V_global = V_global
        self.V_local = V_local
        self.V = V_global + V_local

        if dataset_name == "imagenette":
            self.ds = load_dataset("frgfm/imagenette", "160px", split=split)
        elif dataset_name == "imagenet":
            self.ds = load_dataset("ILSVRC/imagenet-1k", split=split)
        else:
            self.ds = load_dataset(dataset_name, split=split)

        self.global_aug = v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.4, 1.0)),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.local_aug = v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.05, 0.4)),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.test_transform = v2.Compose([
            v2.Resize(img_size),
            v2.CenterCrop(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        label = item["label"]

        if self.V > 1:
            views = [self.global_aug(img) for _ in range(self.V_global)]
            views += [self.local_aug(img) for _ in range(self.V_local)]
            return torch.stack(views), label
        else:
            return torch.stack([self.test_transform(img)]), label

    def __len__(self):
        return len(self.ds)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop — invariance ONLY; collapse prevented by nested PID control
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="pi_jepa")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    wandb.init(project="PI-JEPA", config=OmegaConf.to_container(cfg, resolve=True))
    torch.manual_seed(cfg.get("seed", 0))
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Dataset ──────────────────────────────────────────────────────────
    V_global = cfg.get("V_global", 2)
    V_local = cfg.get("V_local", 6)
    V = V_global + V_local
    img_size = cfg.get("img_size", 128)
    dataset_name = cfg.get("dataset", "imagenette")
    num_classes = cfg.get("num_classes", 10)

    train_ds = MultiCropDataset(
        "train", V_global=V_global, V_local=V_local,
        img_size=img_size, dataset_name=dataset_name,
    )
    test_ds = MultiCropDataset(
        "validation", V_global=0, V_local=0,
        img_size=img_size, dataset_name=dataset_name,
    )
    test_ds.V = 1

    train_loader = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True,
        drop_last=True, num_workers=cfg.get("num_workers", 8), pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, num_workers=cfg.get("num_workers", 8), pin_memory=True,
    )

    # ── Model: backbone selection via token_pid flag ───────────────────
    embed_dim = cfg.get("embed_dim", 512)
    proj_dim = cfg.proj_dim
    use_token_pid = cfg.get("token_pid", True)

    if use_token_pid:
        print("Backbone: PIDformer (token-level PID ON)")
        net = PIDformerEncoder(
            img_size=img_size,
            patch_size=cfg.get("patch_size", 8),
            embed_dim=384,
            depth=cfg.get("depth", 12),
            num_heads=cfg.get("num_heads", 6),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            out_dim=embed_dim,
            proj_dim=proj_dim,
            drop_path_rate=cfg.get("drop_path_rate", 0.1),
            pretrained=cfg.get("pretrained", False),
        ).to(device)
    else:
        model_name = cfg.get("model_name", "vit_small_patch8_224")
        if "resnet" in model_name.lower():
            print("Backbone: ResNet18D")
        else:
            print("Backbone: Standard ViT from timm (token-level PID OFF)")
        net = ViTEncoder(
            model_name=model_name,
            img_size=img_size,
            embed_dim=embed_dim,
            proj_dim=proj_dim,
            drop_path_rate=cfg.get("drop_path_rate", 0.1),
            pretrained=cfg.get("pretrained", False),
        ).to(device)

    # ── Sample-level channel-split PID controller (covariance feedback) ──
    covpid = CovariancePID(
        dim=proj_dim,
        sigma2=cfg.get("sigma2", 1.0),
        k_p=cfg.get("cov_k_p", 1.0),
        k_i=cfg.get("cov_k_i", 0.3),
        k_d=cfg.get("cov_k_d", 0.05),
        eta_i=cfg.get("cov_eta_i", 0.02),
        w_clip=cfg.get("cov_w_clip", 10.0),
        cov_lambda=cfg.get("cov_lambda", 1.0),
        normalize=cfg.get("cov_normalize", True),
        use_integral=cfg.get("cov_use_integral", True),
        use_derivative=cfg.get("cov_use_derivative", True),
        d_filter=cfg.get("cov_d_filter", 1.0),
    ).to(device)

    probe = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)).to(device)

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])

    warmup_steps = len(train_loader)
    total_steps = len(train_loader) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=(device == "cuda"))

    # ── Resume ───────────────────────────────────────────────────────────
    best_acc = 0.0
    start_epoch = 0
    resume_path = cfg.get("resume")
    if resume_path:
        ckpt = torch.load(resume_path, map_location=device, weights_only=False)
        net.load_state_dict(ckpt["net"])
        probe.load_state_dict(ckpt["probe"])
        covpid.load_state_dict(ckpt["covpid"])
        opt.load_state_dict(ckpt["optimizer"])
        best_acc = ckpt.get("best_acc", 0.0)
        start_epoch = ckpt.get("epoch", -1) + 1
        print(f"Resumed from {resume_path}: epoch {start_epoch}, best_acc={best_acc*100:.2f}%")

    ckpt_dir = os.path.join("checkpoints", cfg.get("model_name", "default"))

    # ── Training ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, cfg.epochs):
        net.train()
        probe.train()
        covpid.train()
        for vs, y in tqdm.tqdm(train_loader, total=len(train_loader)):
            vs = vs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device, dtype=torch.bfloat16):
                emb, proj = net(vs)            # emb: (N*V, embed_dim), proj: (V, N, proj_dim)

            # ── Sample-level PID + losses in fp32 (linear algebra stability) ──
            proj = proj.float()
            Vv, Nn, P = proj.shape
            proj_flat = proj.reshape(Vv * Nn, P)
            # Closed-loop control: returns L2-normalized features + covariance control force
            proj_norm_flat, ctrl_loss, ctrl = covpid(proj_flat)
            proj_norm = proj_norm_flat.reshape(Vv, Nn, P)

            # Invariance loss (on normalized features) + PID covariance control force
            inv_loss = (proj_norm.mean(0) - proj_norm).square().mean()

            # Online linear probe (monitors representation quality)
            y_rep = y.repeat_interleave(V)
            yhat = probe(emb.detach().float())
            probe_loss = F.cross_entropy(yhat, y_rep)

            loss = inv_loss + ctrl_loss + probe_loss

            # Collapse diagnostics (on the RAW, uncontrolled features)
            batch_var = emb.detach().float().var(dim=0).mean()        # representation spread
            proj_var = proj_flat.detach().var(dim=0).mean()           # projection spread

            # ── Gradient-flow probe ──
            # ‖∂ctrl_loss/∂proj‖ = how much closed-loop control force actually
            # reaches the encoder. If this were ~0 the controller would be a
            # stop-gradient no-op; it is not. Only backprops through the covariance
            # ops (not the backbone), so it is cheap. retain_graph keeps the graph
            # alive for the main backward below.
            ctrl_grad_norm = torch.autograd.grad(
                ctrl_loss, proj_flat, retain_graph=True
            )[0].norm().item()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            log = {
                "train/inv_loss": inv_loss.item(),
                "train/ctrl_loss": ctrl_loss.item(),
                "train/probe_loss": probe_loss.item(),
                "train/total_loss": loss.item(),
                "train/batch_var": batch_var.item(),
                "train/proj_var": proj_var.item(),
                "grad/ctrl_wrt_proj": ctrl_grad_norm,
            }
            log.update(ctrl)
            wandb.log(log)

        # ── Evaluation ───────────────────────────────────────────────────
        net.eval()
        probe.eval()
        covpid.eval()
        correct = 0
        with torch.inference_mode():
            for vs, y in test_loader:
                vs = vs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast(device, dtype=torch.bfloat16):
                    emb, _ = net(vs)
                    correct += (probe(emb).argmax(1) == y).sum().item()
        acc = correct / len(test_ds)
        best_acc = max(best_acc, acc)
        wandb.log({"test/acc": acc, "test/best_acc": best_acc, "test/epoch": epoch})
        print(f"Epoch {epoch+1}: acc={acc*100:.2f}% (best={best_acc*100:.2f}%)")

        if (epoch + 1) % cfg.get("save_every", 100) == 0 or acc == best_acc:
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save({
                "epoch": epoch, "net": net.state_dict(), "probe": probe.state_dict(),
                "covpid": covpid.state_dict(), "optimizer": opt.state_dict(),
                "best_acc": best_acc,
            }, f"{ckpt_dir}/pi_jepa_epoch{epoch+1}_acc{acc*100:.1f}.pt")

    wandb.finish()
    print(f"\nTraining complete. Best accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
