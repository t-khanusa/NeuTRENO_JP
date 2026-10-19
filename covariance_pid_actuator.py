"""Batch-covariance PID actuator — forward-pass whitening via Newton-Schulz.

Replaces Cholesky/SVD with NS matrix-function iterations:
  z̃ → z̃ @ NS_invsqrt(C + εI) @ NS_sqrt(S_eff)
where S_eff = R + (K_i·W + K_d·dE) / K_p  (PID-corrected target covariance).

Shared map on batch features → invariance-safe for multi-view JEPA.
No anti-collapse loss term; control is entirely in the forward pass.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Muon / MuonSSM quintic coefficients (Jordan et al., 2024)
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315
NS_EPS = 1e-7


def _sym(a: torch.Tensor) -> torch.Tensor:
    return 0.5 * (a + a.transpose(-1, -2))


def _project_psd(m: torch.Tensor, eps: float) -> torch.Tensor:
    """Symmetrize and clamp eigenvalues to keep S_eff valid for NS sqrt."""
    m = _sym(m)
    evals, evecs = torch.linalg.eigh(m)
    evals = evals.clamp_min(eps)
    return evecs @ torch.diag(evals) @ evecs.transpose(-1, -2)


def newton_schulz5(
    g: torch.Tensor,
    steps: int = 5,
    eps: float = NS_EPS,
    a: float = NS_A,
    b: float = NS_B,
    c: float = NS_C,
    use_bfloat16: bool = False,
) -> torch.Tensor:
    """Muon degree-5 NS: approximate polar factor (zeroth matrix power).

    Matches the PyTorch Muon reference (Jordan et al., 2024):
      X /= ||X||_F;  A = X X^T;  X <- a X + (b A + c A^2) X

    Input G (m, n); output same shape. This is NOT C^{-1/2} by itself.
    """
    if g.ndim != 2:
        raise ValueError("newton_schulz5 expects a 2D matrix")
    dtype_out = g.dtype
    x = g.bfloat16() if use_bfloat16 else g.float()
    transposed = x.size(0) > x.size(1)
    if transposed:
        x = x.T
    x = x / x.norm().clamp(min=eps)
    for _ in range(steps):
        gram = x @ x.T
        gram_upd = b * gram + c * (gram @ gram)
        x = a * x + gram_upd @ x
    if transposed:
        x = x.T
    return x.to(dtype=dtype_out)


def _ns_invsqrt_step(
    y: torch.Tensor,
    b_mat: torch.Tensor,
    eye: torch.Tensor,
    *,
    poly: str,
    a: float,
    b: float,
    c_coef: float,
) -> torch.Tensor:
    """One Newton–Schulz step toward B^{-1/2} with Y @ B @ Y ≈ I."""
    m = b_mat @ y @ y
    if poly == "cubic":
        y = y @ (0.5 * (3.0 * eye - m))
    elif poly == "quintic":
        mm = m @ m
        y = y @ ((35.0 * eye - 35.0 * m + 21.0 * mm - 5.0 * (mm @ m)) / 16.0)
    elif poly == "muon5":
        # Muon (a,b,c) as right-multiplier polynomial on M = B Y^2
        y = y @ (a * eye + b * m + c_coef * (m @ m))
    else:
        raise ValueError(f"unknown ns invsqrt poly: {poly}")
    return _sym(y)


def ns_matrix_invsqrt(
    mat: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-6,
    *,
    poly: str = "muon5",
    a: float = NS_A,
    b: float = NS_B,
    c_coef: float = NS_C,
) -> torch.Tensor:
    """Newton-Schulz iteration for C^{-1/2} on SPD matrix C (matmul-only).

    Scale C = t·B with ||B||_2 = 1, iterate on B, then C^{-1/2} = B^{-1/2} / sqrt(t).

    poly:
      - ``muon5``: degree-5 right-multiplier with Muon coeffs (a,b,c) on M = B Y^2
      - ``quintic``: 5th-order inverse-sqrt polynomial (best accuracy at 5 steps)
      - ``cubic``: classic 3rd-order inverse-sqrt iteration
    """
    mat = _sym(mat)
    d = mat.shape[-1]
    eye = torch.eye(d, device=mat.device, dtype=mat.dtype)
    c_reg = mat + eps * eye
    t = torch.linalg.norm(c_reg, ord=2).clamp_min(eps)
    b_mat = c_reg / t
    y = eye
    for _ in range(steps):
        y = _ns_invsqrt_step(y, b_mat, eye, poly=poly, a=a, b=b, c_coef=c_coef)
    return y / torch.sqrt(t)


def ns_matrix_sqrt(
    mat: torch.Tensor,
    steps: int = 5,
    eps: float = 1e-6,
    *,
    poly: str = "muon5",
    a: float = NS_A,
    b: float = NS_B,
    c_coef: float = NS_C,
) -> torch.Tensor:
    """Matrix square root via S @ S^{-1/2} (reuses NS inv-sqrt, same matmul cost)."""
    mat = _project_psd(mat, eps)
    return mat @ ns_matrix_invsqrt(mat, steps=steps, eps=eps, poly=poly, a=a, b=b, c_coef=c_coef)


class CovariancePIDActuator(nn.Module):
    """Sample-level PID with NS whitening actuator (architecture, no ctrl loss).

    Actuator (Option B — replaces Cholesky):
      1. C = Cov(z̃)
      2. E = R - C,  update integral W and derivative dE  [no_grad state]
      3. S_eff = R + (K_i·W + K_d·dE) / K_p
      4. z_ctrl = z̃ @ NS_invsqrt(C) @ NS_sqrt(S_eff)
      5. z_out = μ + (1 - blend)·z̃ + blend·z_ctrl
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
        ns_steps: int = 5,
        cov_eps: float = 1e-4,
        actuator_blend: float | None = None,
        ns_eps: float = NS_EPS,
        ns_a: float = NS_A,
        ns_b: float = NS_B,
        ns_c: float = NS_C,
        ns_poly: str = "muon5",
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
        self.ns_steps = ns_steps
        self.cov_eps = cov_eps
        self.ns_eps = ns_eps
        self.ns_a = ns_a
        self.ns_b = ns_b
        self.ns_c = ns_c
        self.ns_poly = ns_poly
        # blend=1 → full NS whitening path; defaults to k_p for backward compat
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
            # Inference: use frozen integral state (no update)
            s_eff = r + self.k_i * self.W / max(self.k_p, 1e-6)

        ns_kw = dict(
            steps=self.ns_steps,
            eps=self.cov_eps,
            poly=self.ns_poly,
            a=self.ns_a,
            b=self.ns_b,
            c_coef=self.ns_c,
        )

        # ── NS whitening actuator (differentiable w.r.t. z) ──
        c_reg = _sym(c) + self.cov_eps * eye
        w_inv = ns_matrix_invsqrt(c_reg, **ns_kw)
        s_psd = _project_psd(s_eff, self.cov_eps)
        w_sqrt = ns_matrix_sqrt(s_psd, **ns_kw)

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
