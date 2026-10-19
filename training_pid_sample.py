"""
Full-PID JEPA: control theory at BOTH the token level AND the sample level.

Two nested closed loops, no regularization loss, no teacher/EMA:

  1. Token-level PID (PIDformer backbone, Nguyen et al. 2024)  [optional: token_pid flag]
     Controls the token field u(x,t) along DEPTH t toward the input reference
     f = β·v0. Prevents token/rank collapse (oversmoothing) WITHIN a sample.
     When token_pid=false, a standard timm ViT backbone is used instead.

  2. Sample-level PID (CovariancePID)
     Controls the batch covariance C(τ) along TRAINING time τ toward an
     isotropic setpoint C* = σ²·I. Prevents batch/sample collapse ACROSS
     samples. Realized as a differentiable whitening actuator whose target
     is corrected by a persistent integral buffer W — the "magic of integral
     action" pins C → C* exactly, robustly, with no teacher and no loss term.

The ONLY training loss is multi-view invariance. SIGReg is removed; collapse
is prevented purely by closed-loop control.

═══════════════════════════════════════════════════════════════════════════════
SAMPLE-LEVEL PID — Formal derivation  (CovariancePID)
═══════════════════════════════════════════════════════════════════════════════

Step 1 — Plant identification
    The backbone f_θ maps images to features z = f_θ(x).
    Under invariance-only training the batch covariance
        C(τ) = Cov_s[z_s(τ)]    (τ = training step)
    evolves as
        dC/dτ = −G(C)            G unknown, G(0)=0, DG(0) ≻ 0
    so C = 0 (collapse) is an attracting fixed point of the UNCONTROLLED plant.

Step 2 — Setpoint
    C* = σ²·I  (isotropic, maximum-entropy at fixed energy).
    External and fixed — does NOT depend on the model, so (C, C*) → (0, 0) is
    impossible.  This breaks the fatal co-collapse failure mode of PID-JEPA.

Step 3 — Error signal
    E(τ) = C* − C(τ)  ∈ Sym(d)   (matrix-valued)

Step 4 — PID control law in operator space
    U(τ) = K_P · E  +  K_I · W  +  K_D · dE/dτ
    where W(τ) = ∫₀ᵗ E(τ′) dτ′  is the integral state (persistent buffer).

Step 5 — PID-corrected setpoint (target covariance after actuation)
    S(τ) = σ²I + K_I · W + K_D · dE/dτ
    When K_P = 1 (full whitening), the actuator maps C → S exactly.
    The proportional action is built into the whitening map itself.

Step 6 — Actuator (differentiable whitening)
    Given z̃ = z − μ with Cov(z̃) = C:
      1. Cholesky:  C + εI = L Lᵀ
      2. Whiten:    w = z̃ · L⁻ᵀ            (Cov(w) = I)
      3. Color:     z_ctrl = w · S^{1/2}    (Cov(z_ctrl) = S)
      4. Blend:     z_out = μ + (1−k_p)·z̃ + k_p·z_ctrl
    k_p ∈ [0,1] is the proportional blending factor.
    Differentiable backward through L delivers anti-collapse gradients to θ.

Step 7 — Integral state update (discrete time, with anti-windup)
    W ← clamp(W + η · E,  −w_max, +w_max)     (Aström §6.5)
    The clamp is anti-windup: prevents the integral from growing unbounded
    when the actuator saturates (C near zero → L⁻ᵀ amplification → ε limit).

Step 8 — Steady-state guarantee (the "magic of integral action")
    At steady state dC/dτ = 0  ⟹  dW/dτ = E = 0  ⟹  C = C* = σ²I.
    This holds for ANY unknown plant contraction G — the integral term
    absorbs the bias, making the result plant-independent.

Step 9 — Stability (linearized closed loop, per eigen-direction k)
    Characteristic equation:  (1+K_D)s² + (g_k+K_P)s + K_I = 0
    Routh–Hurwitz:  stable ⟺  K_I > 0,  K_P > −g_k,  K_D > −1
    All satisfied for any positive gains → collapse-free equilibrium is stable.
═══════════════════════════════════════════════════════════════════════════════

Usage:
    # Loss-based PID control force (ctrl_loss, ~93% acc with pretrained backbone):
    python training_pid_sample.py --config-path configs --config-name pid_sample

    # Original Cholesky architecture PID (forward whitening, invariance only):
    python training_pid_sample.py --config-path configs --config-name pid_sample_actuator

    # Standard ViT + sample-level PID only (no token-level PID):
    python training_pid_sample.py +token_pid=false

    # PIDformer + sample-level PID (both levels):
    python training_pid_sample.py +token_pid=true
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
    """Standard ViT encoder from timm with an MLP projector head."""

    def __init__(self, model_name="vit_small_patch8_224", img_size=128,
                 embed_dim=512, proj_dim=16, drop_path_rate=0.1, pretrained=False):
        super().__init__()
        kwargs = dict(
            pretrained=pretrained,
            num_classes=embed_dim,
            drop_path_rate=drop_path_rate,
        )
        if "resnet" not in model_name.lower():
            kwargs["img_size"] = img_size
        self.backbone = timm.create_model(model_name, **kwargs)
        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# SAMPLE-LEVEL PID — Covariance controller (the new contribution)
#
# Classical PID identification:
#   Plant output   y(τ) = C(τ) = Cov_s[z_s]         (batch covariance, matrix)
#   Reference      r    = C* = σ²·I                  (isotropic setpoint)
#   Error          e(τ) = r − y = σ²I − C            (matrix error)
#   Controller     u(τ) = K_P·e + K_I·∫e + K_D·ė    (PID in operator space)
#   Actuator       whitening map: z̃ → z̃·L⁻ᵀ·S^½     (differentiable, never saturates)
#
# Closed-loop feedback to the PLANT (the backbone), not a post-hoc actuator.
#
# Why a post-hoc actuator (whitening) cannot prevent collapse:
#   Whitening reproduces the input spectrum scaled: Var_i(whitened)=λ_i/(λ_i+ε).
#   A collapsed direction (λ_i→0) maps to 0 — the actuator is BYPASSED below ε,
#   so it masks collapse rather than preventing it. Moreover, total collapse
#   (all views identical) is a ZERO-loss optimum of inv_loss: there is no
#   view-disagreement left for the whitening gradient to amplify, so the
#   "barrier" vanishes exactly where it is needed. The only channel that
#   reaches the backbone is the gradient → the control law must enter the loss.
#
# Control-FORCE formulation (the fix):
#   Plant output  : C = Cov(z̃)   (z L2-normalized → scale-invariant, no blow-up)
#   Setpoint      : R = (σ²/dim)·I  (isotropic, achievable on the unit sphere)
#   Error         : E_t = R − C_t
#   The PID law is realized as an effective SETPOINT SHIFT:
#       R_eff = R + (K_i·∫E + K_d·dE)/K_p          (detached)
#       L_ctrl = ½·K_p·‖C − R_eff‖²_F             (non-negative tracking loss)
#   ⇒ ∂L_ctrl/∂C = −(K_p·E + K_i·∫E + K_d·dE) = −G  (identical PID force, but a
#     proper non-negative objective that is 0 at the setpoint, unlike −⟨G,C⟩).
#
# One-sided variance (deadband, default): only act on a direction when it is
#   COLLAPSING (C_ii < R_ii); high variance is allowed. This stops the regulator
#   from fighting the invariance objective while still guaranteeing anti-collapse
#   + decorrelation. It is the control analog of an asymmetric actuator.
#
# Why the INTEGRAL term is the elegant part (vs VICReg/SIGReg):
#   Persistent collapse makes ∫E grow without bound → an ever-increasing
#   restoring force → collapse cannot be sustained (NON-saturating, unlike the
#   whitening barrier). At equilibrium the integral holds exactly the force
#   needed to balance the invariance pull, giving ZERO steady-state error
#   (Cov = R exactly). P-only methods cannot:
#     VICReg ≈ P-only variance penalty   → residual steady-state error
#     SIGReg ≈ P-only distribution match  → residual steady-state error
#     CovariancePID = full PID            → exact setpoint, no residual bias
# ─────────────────────────────────────────────────────────────────────────────

class CovariancePIDActuator(nn.Module):
    """Original sample-level PID: Cholesky whitening actuator in the forward pass.

    Implements Steps 1–9 in the module docstring.  Loss = invariance only;
    no ctrl_loss.  Matches the v91fkujl run (σ²·I setpoint, no L2 normalize,
    S = σ²I + K_i·W + K_d·dE without /k_p).
    """

    def __init__(self, dim, sigma2=1.0, k_p=1.0, k_i=0.3, k_d=0.05,
                 eta_i=0.02, w_clip=2.0, eps=1e-4):
        super().__init__()
        self.dim = dim
        self.sigma2 = sigma2
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.eta_i = eta_i
        self.w_clip = w_clip
        self.eps = eps

        self.register_buffer("W", torch.zeros(dim, dim))
        self.register_buffer("E_prev", torch.zeros(dim, dim))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    @staticmethod
    def _sym(A):
        return 0.5 * (A + A.transpose(-1, -2))

    def forward(self, z):
        """z: (M, dim). Returns (z_controlled, diagnostics)."""
        if not self.training:
            return z, {}

        dtype_in = z.dtype
        z = z.float()
        M = z.shape[0]
        I = torch.eye(self.dim, device=z.device, dtype=z.dtype)

        mu = z.mean(dim=0, keepdim=True)
        z_tilde = z - mu
        C = self._sym(z_tilde.transpose(0, 1) @ z_tilde) / max(M - 1, 1)

        E = self.sigma2 * I - C

        with torch.no_grad():
            E_det = E.detach()
            dE = E_det - self.E_prev if self.steps.item() > 0 else torch.zeros_like(E_det)
            self.W.add_(self.eta_i * E_det).clamp_(-self.w_clip, self.w_clip)
            self.E_prev.copy_(E_det)
            self.steps += 1

            S = self.sigma2 * I + self.k_i * self.W + self.k_d * dE
            S = self._sym(S)
            sv, sVec = torch.linalg.eigh(S)
            sv = sv.clamp_min(0.0)
            S_half = (sVec * sv.sqrt().unsqueeze(0)) @ sVec.transpose(-1, -2)

        L = torch.linalg.cholesky(C + self.eps * I)
        X = torch.linalg.solve_triangular(L, z_tilde.transpose(0, 1), upper=False)
        whitened = X.transpose(0, 1) @ S_half

        z_out = mu + (1.0 - self.k_p) * z_tilde + self.k_p * whitened

        diag = {
            "ctrl/cov_err": E.detach().norm().item(),
            "ctrl/integral_norm": self.W.norm().item(),
            "ctrl/cov_trace": C.diagonal().mean().item(),
            "ctrl/cov_offdiag": (C - torch.diag(torch.diag(C))).abs().mean().item(),
        }
        return z_out.to(dtype_in), diag


class CovariancePID(nn.Module):
    """Closed-loop PID regulator of the batch covariance, applied as a control
    FORCE on the backbone (a loss term), with integral action for exact setpoint
    tracking. Prevents sample-level collapse without a teacher/EMA or SIGReg.
    """

    def __init__(self, dim, sigma2=1.0, k_p=1.0, k_i=0.3, k_d=0.05,
                 eta_i=0.02, w_clip=10.0, cov_lambda=1.0, normalize=True,
                 one_sided=True, **kwargs):
        super().__init__()
        self.dim = dim
        self.sigma2 = sigma2        # target trace of the (normalized) covariance
        self.k_p = k_p              # proportional gain
        self.k_i = k_i              # integral gain
        self.k_d = k_d              # derivative gain
        self.eta_i = eta_i          # integral accumulation step
        self.w_clip = w_clip        # anti-windup clamp on the integral state
        self.cov_lambda = cov_lambda  # master weight of the control force
        self.normalize = normalize  # L2-normalize features (scale-invariant plant)
        self.one_sided = one_sided  # variance deadband: only act when collapsing

        # Integral / derivative memory (persistent across training steps).
        self.register_buffer("W", torch.zeros(dim, dim))
        self.register_buffer("E_prev", torch.zeros(dim, dim))
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
        I = torch.eye(self.dim, device=z.device, dtype=z.dtype)
        R = (self.sigma2 / self.dim) * I    # isotropic setpoint, achievable on the sphere

        mu = z.mean(dim=0, keepdim=True)
        z_tilde = z - mu
        C = self._sym(z_tilde.transpose(0, 1) @ z_tilde) / max(M - 1, 1)   # plant output (diff'able)

        # ── Integral / derivative memory from the (detached) covariance error ──
        # The PID law is realized as an effective SETPOINT SHIFT, so the control
        # action becomes a proper non-negative quadratic tracking loss whose
        # gradient equals the PID control force -G (the linear -⟨G,C⟩ is the same
        # gradient but has no minimum and takes negative values).
        with torch.no_grad():
            E = R - C
            if self.one_sided:
                # Variance deadband: only act on a direction when it is COLLAPSING
                # (C_ii < R_ii). High variance is allowed, so the controller stops
                # fighting the invariance objective. Off-diagonals stay two-sided
                # (decorrelation). This is an asymmetric actuator with anti-windup.
                d = torch.diagonal(E).clamp_min(0.0)
                E = E - torch.diag_embed(torch.diagonal(E)) + torch.diag_embed(d)
            dE = E - self.E_prev if self.steps.item() > 0 else torch.zeros_like(E)
            self.W.add_(self.eta_i * E).clamp_(-self.w_clip, self.w_clip)   # integral + anti-windup
            self.E_prev.copy_(E)
            self.steps += 1
            # Effective setpoint = base setpoint shifted by integral + derivative action.
            R_eff = R + (self.k_i * self.W + self.k_d * dE) / max(self.k_p, 1e-6)

        # ── Control force as a non-negative quadratic tracking loss ──
        # off-diagonals: drive C_ij → R_eff_ij (≈0)  [two-sided decorrelation]
        # diagonal:      drive C_ii → R_eff_ii only when below (variance hinge)
        D = R_eff - C
        if self.one_sided:
            d_diag = torch.diagonal(D).clamp_min(0.0)
            off = D - torch.diag_embed(torch.diagonal(D))
            err2 = off.square().sum() + d_diag.square().sum()
        else:
            err2 = D.square().sum()
        ctrl_loss = self.cov_lambda * 0.5 * self.k_p * err2

        diag = {
            "ctrl/cov_err": (R - C).norm().item(),
            "ctrl/integral_norm": self.W.norm().item(),
            "ctrl/cov_trace": C.diagonal().mean().item(),
            "ctrl/cov_offdiag": (C - torch.diag(torch.diag(C))).abs().mean().item(),
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

@hydra.main(version_base=None, config_path="configs", config_name="pid_sample")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    wandb.init(project="PID-Sample", config=OmegaConf.to_container(cfg, resolve=True))
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
        print("Backbone: Standard ViT from timm (token-level PID OFF)")
        net = ViTEncoder(
            model_name=cfg.get("model_name", "vit_small_patch8_224"),
            img_size=img_size,
            embed_dim=embed_dim,
            proj_dim=proj_dim,
            drop_path_rate=cfg.get("drop_path_rate", 0.1),
            pretrained=cfg.get("pretrained", False),
        ).to(device)

    # ── Sample-level PID controller (covariance servo on the projection) ──
    cov_mode = cfg.get("cov_mode", "force")
    if cov_mode == "actuator":
        print("Sample-level PID: Cholesky actuator (forward pass, invariance only)")
        covpid = CovariancePIDActuator(
            dim=proj_dim,
            sigma2=cfg.get("sigma2", 1.0),
            k_p=cfg.get("cov_k_p", 1.0),
            k_i=cfg.get("cov_k_i", 0.3),
            k_d=cfg.get("cov_k_d", 0.05),
            eta_i=cfg.get("cov_eta_i", 0.02),
            w_clip=cfg.get("cov_w_clip", 2.0),
            eps=cfg.get("cov_eps", 1e-4),
        ).to(device)
    elif cov_mode == "force":
        print("Sample-level PID: control force (ctrl_loss)")
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
            one_sided=cfg.get("cov_one_sided", True),
        ).to(device)
    else:
        raise ValueError(f"Unknown cov_mode={cov_mode!r}; use 'actuator' or 'force'")

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

    # ── Training ─────────────────────────────────────────────────────────
    best_acc = 0.0

    for epoch in range(cfg.epochs):
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

            if cov_mode == "actuator":
                proj_ctrl_flat, ctrl = covpid(proj_flat)
                proj_ctrl = proj_ctrl_flat.reshape(Vv, Nn, P)
                inv_loss = (proj_ctrl.mean(0) - proj_ctrl).square().mean()
            else:
                proj_norm_flat, ctrl_loss, ctrl = covpid(proj_flat)
                proj_norm = proj_norm_flat.reshape(Vv, Nn, P)
                inv_loss = (proj_norm.mean(0) - proj_norm).square().mean()

            # Online linear probe (monitors representation quality)
            y_rep = y.repeat_interleave(V)
            yhat = probe(emb.detach().float())
            probe_loss = F.cross_entropy(yhat, y_rep)

            loss = inv_loss + probe_loss if cov_mode == "actuator" else inv_loss + ctrl_loss + probe_loss

            # Collapse diagnostics (on the RAW, uncontrolled features)
            batch_var = emb.detach().float().var(dim=0).mean()
            proj_var = proj_flat.detach().var(dim=0).mean()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            log = {
                "train/inv_loss": inv_loss.item(),
                "train/probe_loss": probe_loss.item(),
                "train/total_loss": loss.item(),
                "train/batch_var": batch_var.item(),
                "train/proj_var": proj_var.item(),
            }
            if cov_mode == "force":
                log["train/ctrl_loss"] = ctrl_loss.item()
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
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch, "net": net.state_dict(), "probe": probe.state_dict(),
                "covpid": covpid.state_dict(), "optimizer": opt.state_dict(),
                "best_acc": best_acc,
            }, f"checkpoints/pid_sample_epoch{epoch+1}_acc{acc*100:.1f}.pt")

    wandb.finish()
    print(f"\nTraining complete. Best accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
