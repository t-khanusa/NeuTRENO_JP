"""Batch-covariance PID actuator — forward-pass Cholesky whitening.

Sample-level PID (architecture, no ctrl_loss):
  z̃ → z̃ @ L^{-T} @ S^{1/2}
where C + εI = L Lᵀ,  S = R + (K_i·W + K_d·dE) / K_p  (PID-corrected target).

Shared map on batch features → invariance-safe for multi-view JEPA.
The ONLY anti-collapse mechanism is the closed-loop actuator in the forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _sym(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + a.transpose(-1, -2))


def _project_psd(m: torch.Tensor, eps: float) -> torch.Tensor:
    """Symmetrize and clamp eigenvalues so S is valid for matrix sqrt."""
    m = _sym(m)
    evals, evecs = torch.linalg.eigh(m)
    evals = evals.clamp_min(eps)
    return evecs @ torch.diag(evals) @ evecs.transpose(-1, -2)


def matrix_sqrt_psd(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Exact S^{1/2} for SPD matrix S via eigh (differentiable)."""
    mat = _project_psd(mat, eps)
    evals, evecs = torch.linalg.eigh(mat)
    return evecs @ torch.diag(evals.sqrt()) @ evecs.transpose(-1, -2)


def cholesky_invsqrt(mat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """C^{-1/2} from Cholesky: C + εI = L Lᵀ  →  L^{-T}.

    Falls back to eigh-based inverse sqrt if Cholesky fails (near-singular batch).
    """
    mat = _sym(mat)
    d = mat.shape[-1]
    eye = torch.eye(d, device=mat.device, dtype=mat.dtype)
    c_reg = mat + eps * eye
    try:
        l = torch.linalg.cholesky(c_reg, upper=False)
        # z @ L^{-T}  whitens rows when Cov(z) ≈ C
        l_inv = torch.linalg.solve_triangular(l, eye, upper=False)
        return l_inv.transpose(-1, -2)
    except torch.linalg.LinAlgError:
        evals, evecs = torch.linalg.eigh(c_reg)
        inv_sqrt = evals.clamp_min(eps).rsqrt()
        return evecs @ torch.diag(inv_sqrt) @ evecs.transpose(-1, -2)


class CovariancePIDActuatorCholesky(nn.Module):
    """Sample-level PID with Cholesky whitening actuator (architecture, no ctrl loss).

    Steps 1–9 (formal derivation):
      1. Plant: C(τ) = Cov(z̃)
      2. Setpoint: R = (σ²/d) I
      3. Error: E = R − C
      4–5. S_eff = R + (K_i·W + K_d·dE) / K_p
      6. z_ctrl = z̃ @ C^{-1/2} @ S_eff^{1/2}
         z_out = μ + (1 − blend)·z̃ + blend·z_ctrl
      7. W ← clamp(W + η·E)   [no_grad, anti-windup]
    """

    def __init__(
        self,
        dim: int,
        sigma2: float = 1.0,
        k_p: float = 1.0,
        k_i: float = 0.3,
        k_d: float = 0.05,
        eta_i: float = 0.02,
        w_clip: float = 10.0,
        normalize: bool = True,
        one_sided: bool = True,
        cov_eps: float = 1e-4,
        actuator_blend: float | None = None,
    ):
        super().__init__()
        self.dim = dim
        self.sigma2 = sigma2
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.eta_i = eta_i
        self.w_clip = w_clip
        self.normalize = normalize
        self.one_sided = one_sided
        self.cov_eps = cov_eps
        self.actuator_blend = k_p if actuator_blend is None else actuator_blend

        self.register_buffer("W", torch.zeros(dim, dim))
        self.register_buffer("E_prev", torch.zeros(dim, dim))
        self.register_buffer("steps", torch.zeros((), dtype=torch.long))

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """z: (M, dim). Returns (z_controlled, diagnostics)."""
        dtype_in = z.dtype
        z = z.float()
        if self.normalize:
            z = F.normalize(z, dim=-1, eps=1e-6)

        m, d = z.shape
        eye = torch.eye(d, device=z.device, dtype=z.dtype)
        r = (self.sigma2 / d) * eye

        mu = z.mean(dim=0, keepdim=True)
        z_tilde = z - mu
        c = _sym(z_tilde.transpose(0, 1) @ z_tilde) / max(m - 1, 1)

        # ── PID state (persistent across steps, detached from autograd) ──
        if self.training:
            with torch.no_grad():
                e = r - c
                if self.one_sided:
                    diag_e = torch.diagonal(e).clamp_min(0.0)
                    e = e - torch.diag_embed(torch.diagonal(e)) + torch.diag_embed(diag_e)
                de = e - self.E_prev if self.steps.item() > 0 else torch.zeros_like(e)
                self.W.add_(self.eta_i * e).clamp_(-self.w_clip, self.w_clip)
                self.E_prev.copy_(e)
                self.steps += 1
                s_eff = r + (self.k_i * self.W + self.k_d * de) / max(self.k_p, 1e-6)
        else:
            s_eff = r + self.k_i * self.W / max(self.k_p, 1e-6)

        # ── Cholesky whitening actuator (differentiable w.r.t. z) ──
        c_reg = _sym(c) + self.cov_eps * eye
        w_inv = cholesky_invsqrt(c_reg, eps=self.cov_eps)
        s_psd = _project_psd(s_eff, self.cov_eps)
        w_sqrt = matrix_sqrt_psd(s_psd, eps=self.cov_eps)

        z_ctrl = z_tilde @ w_inv @ w_sqrt
        blend = float(max(0.0, min(1.0, self.actuator_blend)))
        z_out = mu + (1.0 - blend) * z_tilde + blend * z_ctrl

        with torch.no_grad():
            zc_out = z_out - z_out.mean(dim=0, keepdim=True)
            c_out = _sym(zc_out.transpose(0, 1) @ zc_out) / max(m - 1, 1)
            c_diag = c_out.double().cpu() + self.cov_eps * torch.eye(d).double()
            try:
                evals = torch.linalg.eigvalsh(c_diag).clamp_min(0.0).float()
            except torch.linalg.LinAlgError:
                evals = torch.diagonal(c_out).clamp_min(0.0)

        diag = {
            "ctrl/cov_err": (r - c).norm().item(),
            "ctrl/cov_err_out": (r - c_out).norm().item(),
            "ctrl/integral_norm": self.W.norm().item(),
            "ctrl/cov_trace": c.diagonal().mean().item(),
            "ctrl/cov_trace_out": c_out.diagonal().mean().item(),
            "ctrl/cov_offdiag": (c - torch.diag(c.diag())).abs().mean().item(),
            "ctrl/min_eval_out": evals.min().item(),
            "ctrl/erank_out": (evals.sum().square() / evals.square().sum().clamp_min(1e-12)).item(),
        }
        return z_out.to(dtype_in), diag
