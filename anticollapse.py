"""
Anti-collapse regularizers behind one interface, so the NeuTRENO backbone can be
crossed with each of them in training_neutreno_reg.py.

Every regularizer receives
    proj : (V, N, P)   projector output, view-major (as returned by the encoders)
    emb  : (V, N, E)   backbone embedding (the probed representation, "h")
and returns (repr_loss, terms) where repr_loss already contains the invariance
term weighted the way the original method weights it.

    sigreg      LeJEPA (Balestriero & LeCun, 2511.08544)    ECF matching to N(0, I)
    visreg      VISReg (Wu et al., 2606.02572)              center + scale + sliced W2 shape
    lambdajepa  λ-JEPA (Demirel et al., 2609.35288)         SACReg at z and at h
    sacreg      λ-JEPA without the backbone term            SACReg at z only
    vicreg      VICReg (Bardes et al., 2022)                variance hinge + covariance
    barlow      Barlow Twins (Zbontar et al., 2021)         cross-correlation → I
    uniformity  Wang & Isola (2020)                         Gaussian-potential uniformity on S^{P-1}
    none        invariance only                             (collapses)
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from training_lejepa import SIGReg


def invariance(proj):
    """View-to-mean MSE (the invariance term of every pipeline in this repo)."""
    return (proj.mean(0) - proj).square().mean()


# ─────────────────────────────────────────────────────────────────────────────
# VISReg — as released (per-view batch statistics, 256 slices)
# ─────────────────────────────────────────────────────────────────────────────

class VISRegLoss(nn.Module):
    def __init__(self, num_projections=256, center_weight=1.0, scale_weight=1.0,
                 shape_weight=1.0):
        super().__init__()
        self.K = num_projections
        self.center_weight = center_weight
        self.scale_weight = scale_weight
        self.shape_weight = shape_weight

    def forward(self, z):
        _, B, D = z.shape
        mu = z.mean(dim=1, keepdim=True)
        center = mu.square().mean()
        zc = z - mu
        std = zc.norm(dim=1).div(math.sqrt(B)).clamp_min(1e-6)
        scale = (std - 1.0).square().mean()
        z_norm = zc / std.detach().unsqueeze(1)
        W = F.normalize(torch.randn(D, self.K, device=z.device, dtype=z.dtype), dim=0)
        p_sorted = (z_norm @ W).sort(dim=1).values
        q = torch.linspace(1, B, B, device=z.device, dtype=z.dtype) / (B + 1)
        target = torch.erfinv(2 * q - 1).mul(math.sqrt(2)).view(1, B, 1)
        shape = (p_sorted - target).square().mean()
        total = (self.center_weight * center + self.scale_weight * scale
                 + self.shape_weight * shape)
        return total, {"center": center, "scale": scale, "shape": shape}


# ─────────────────────────────────────────────────────────────────────────────
# SACReg — λ-JEPA's KL(N(mu, Sigma) || N(0, I)) / d on a random orthonormal slice
# ─────────────────────────────────────────────────────────────────────────────

class SACReg(nn.Module):
    def __init__(self, d_slice=128, eps=1e-4, d_draw=None):
        super().__init__()
        self.d_slice, self.eps, self.d_draw = d_slice, eps, d_draw or d_slice

    def forward(self, x):
        x = x.reshape(-1, x.size(-1))
        d = min(self.d_slice, x.size(1))
        Q, _ = torch.linalg.qr(torch.randn(x.size(1), max(self.d_draw, d), device=x.device))
        p = x @ Q[:, :d]
        mu = p.mean(0)
        pc = p - mu
        S = pc.T @ pc / (p.size(0) - 1)
        cov = S + self.eps * torch.eye(d, device=x.device)
        logdet = 2 * torch.linalg.cholesky(cov).diagonal().log().sum()
        return 0.5 * (cov.diagonal().sum() + mu.square().sum() - d - logdet) / d


# ─────────────────────────────────────────────────────────────────────────────
# VICReg / Barlow Twins / uniformity terms
# ─────────────────────────────────────────────────────────────────────────────

def vicreg_variance(z, gamma=1.0, eps=1e-4):
    return F.relu(gamma - (z.var(0) + eps).sqrt()).mean()


def vicreg_covariance(z):
    zc = z - z.mean(0)
    C = (zc.T @ zc) / (z.shape[0] - 1)
    return (C.square().sum() - C.diagonal().square().sum()) / C.shape[0]


def barlow_loss(z1, z2, lambd):
    n = z1.shape[0]
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-5)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-5)
    c = z1.T @ z2 / n
    on = (torch.diagonal(c) - 1).square().sum()
    off = c.square().sum() - torch.diagonal(c).square().sum()
    return on + lambd * off, on, off


def uniformity_loss(z, t=2.0):
    z = F.normalize(z, dim=-1)
    return torch.pdist(z).square().mul(-t).exp().mean().log()


# ─────────────────────────────────────────────────────────────────────────────
# Registry
# ─────────────────────────────────────────────────────────────────────────────

REGS = ("sigreg", "visreg", "lambdajepa", "sacreg", "vicreg", "barlow", "uniformity", "none")


class AntiCollapse(nn.Module):
    """Computes repr_loss = (weighted invariance) + (weighted regularizer) in fp32."""

    def __init__(self, cfg, n_global=2):
        super().__init__()
        self.reg = str(cfg.get("reg", "sigreg"))
        if self.reg not in REGS:
            raise ValueError(f"Unknown reg={self.reg!r}. Choose from {REGS}")
        self.cfg = cfg
        self.n_global = n_global
        if self.reg == "sigreg":
            self.sigreg = SIGReg()
        elif self.reg == "visreg":
            self.visreg = VISRegLoss(num_projections=cfg.get("visreg_slices", 256))
        elif self.reg in ("lambdajepa", "sacreg"):
            self.cond_z = SACReg(d_slice=cfg.get("lj_z_d_slice", 128), d_draw=128)
            self.cond_h = SACReg(d_slice=cfg.get("lj_h_d_slice", 64), d_draw=128)

    def _center(self, x):
        """λ-JEPA multi-crop default: SACReg on per-image view centers (N rows)."""
        return x.mean(0) if self.cfg.get("lj_batch", "center") == "center" else x.flatten(0, 1)

    def forward(self, proj, emb):
        proj = proj.float()
        emb = emb.float()
        inv = invariance(proj)
        cfg = self.cfg
        terms = {"inv": inv}

        if self.reg == "sigreg":
            r = self.sigreg(proj)
            lamb = cfg.get("lamb", 0.02)
            loss = lamb * r + (1 - lamb) * inv
            terms["sigreg"] = r

        elif self.reg == "visreg":
            r, parts = self.visreg(proj)
            lamb = cfg.get("visreg_lamb", 0.6)
            loss = lamb * r + (1 - lamb) * inv
            terms["visreg"] = r
            terms.update({f"visreg_{k}": v for k, v in parts.items()})

        elif self.reg in ("lambdajepa", "sacreg"):
            rz = self.cond_z(self._center(proj))
            loss = cfg.get("lj_w_inv", 25.0) * inv + cfg.get("lj_w_z", 45.0) * rz
            terms["moment_kl_z"] = rz
            if self.reg == "lambdajepa":
                rh = self.cond_h(self._center(emb))
                loss = loss + cfg.get("lj_w_h", 0.65) * rh
                terms["moment_kl_h"] = rh

        elif self.reg == "vicreg":
            # original VICReg sums var/cov over its two views; with V views we average
            # per-view terms and rescale by 2 to keep the same magnitude.
            var = 2 * torch.stack([vicreg_variance(z) for z in proj]).mean()
            cov = 2 * torch.stack([vicreg_covariance(z) for z in proj]).mean()
            loss = (cfg.get("vic_w_inv", 25.0) * inv + cfg.get("vic_w_var", 25.0) * var
                    + cfg.get("vic_w_cov", 1.0) * cov)
            terms.update({"vic_var": var, "vic_cov": cov})

        elif self.reg == "barlow":
            # every view is correlated against the mean of the global views (multi-crop
            # Barlow); the diagonal term is Barlow's own invariance, `inv` is logged only.
            target = proj[: self.n_global].mean(0)
            lambd = cfg.get("bt_lambda", 5e-3)
            outs = [barlow_loss(z, target, lambd) for z in proj]
            loss = cfg.get("bt_scale", 1.0) * torch.stack([o[0] for o in outs]).mean()
            terms.update({"bt_on": torch.stack([o[1] for o in outs]).mean(),
                          "bt_off": torch.stack([o[2] for o in outs]).mean()})

        elif self.reg == "uniformity":
            # Wang & Isola alignment + uniformity on the hypersphere, per view
            zn = F.normalize(proj, dim=-1)
            align = (zn.mean(0, keepdim=True) - zn).square().sum(-1).mean()
            unif = torch.stack([uniformity_loss(z, cfg.get("unif_t", 2.0)) for z in proj]).mean()
            loss = align + cfg.get("unif_w", 1.0) * unif
            terms.update({"align": align, "unif": unif})

        else:  # none
            loss = inv

        return loss, terms
