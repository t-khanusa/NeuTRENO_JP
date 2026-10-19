#!/usr/bin/env python3
"""
Compare isotropy-inducing transforms on batch embeddings:

  1. SVD whitening          — gold standard (exact Cov -> I when full-rank)
  2. Cholesky whitening     — practical whitening via C + eps I = L L^T
  3. NS degree-5 (Muon)     — zeroth-matrix-power / polar orthogonalization
  4. NS inverse-sqrt (5 it) — Newton-Schulz iteration for C^{-1/2} (whitening family)

Also reports what Muon NS *actually* does to singular values vs true whitening.

Run:
    python test_isotropy_transforms.py
    python test_isotropy_transforms.py --dim 64 --batch 128 --device cuda
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from typing import Callable

import torch

# Muon / MuonSSM quintic coefficients (Jordan et al., 2024)
NS_A, NS_B, NS_C = 3.4445, -4.7750, 2.0315
NS_EPS = 1e-7


# ─────────────────────────────────────────────────────────────────────────────
# Core linear algebra helpers
# ─────────────────────────────────────────────────────────────────────────────

def batch_cov(z: torch.Tensor) -> torch.Tensor:
    """Sample covariance of rows z: (B, d) -> (d, d), unbiased."""
    b, _ = z.shape
    zc = z - z.mean(dim=0, keepdim=True)
    return (zc.transpose(0, 1) @ zc) / max(b - 1, 1)


def isotropy_metrics(z: torch.Tensor, target: float = 1.0) -> dict[str, float]:
    """Lower is better for deviation / anisotropy; erank closer to d is better."""
    c = batch_cov(z)
    d = c.shape[0]
    evals = torch.linalg.eigvalsh(c).clamp_min(0.0)
    mean_e = evals.mean().clamp_min(1e-12)
    off = c - torch.diag(torch.diag(c))
    erank = (evals.sum().square() / evals.square().sum().clamp_min(1e-12)).item()
    return {
        "fro_dev": (c - target * torch.eye(d, device=c.device, dtype=c.dtype)).norm().item()
        / math.sqrt(d),
        "spec_std": (evals.std() / mean_e).item(),
        "aniso_ratio": (evals.max() / evals.min().clamp_min(1e-12)).item(),
        "trace": evals.sum().item(),
        "erank": erank,
        "offdiag": off.abs().mean().item(),
        "min_eval": evals.min().item(),
        "max_eval": evals.max().item(),
    }


def newton_schulz5_zeroth(
    g: torch.Tensor,
    steps: int = 5,
    eps: float = NS_EPS,
    coeffs: tuple[float, float, float] = (NS_A, NS_B, NS_C),
) -> torch.Tensor:
    """Muon / MuonSSM degree-5 NS: approximate polar factor (zeroth matrix power).

    Input G (m, n); output same shape, approx US'V^T with S'_{ii} in ~[0.5, 1.5].
    This is NOT matrix inverse square root and NOT whitening by itself.
    """
    if g.ndim != 2:
        raise ValueError("NS expects a 2D matrix")
    a, b, c = coeffs
    x = g.float()
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
    return x.to(dtype=g.dtype)


def ns_invsqrt_whiten_matrix(c: torch.Tensor, steps: int = 5, eps: float = 1e-6) -> torch.Tensor:
    """Newton-Schulz iteration for C^{-1/2} on SPD matrix C (whitening family).

    Scale C = t·B with ||B||_2 = 1, iterate on B, then C^{-1/2} = B^{-1/2} / sqrt(t).
    """
    d = c.shape[0]
    eye = torch.eye(d, device=c.device, dtype=c.dtype)
    c_reg = c + eps * eye
    t = torch.linalg.norm(c_reg, ord=2).clamp_min(eps)
    b = c_reg / t
    y = eye
    for _ in range(steps):
        y = 0.5 * y @ (3.0 * eye - b @ y @ y)
    return y / torch.sqrt(t)


# ─────────────────────────────────────────────────────────────────────────────
# Transform methods: Z (B, d) -> Z_out (B, d)
# ─────────────────────────────────────────────────────────────────────────────

def svd_whiten(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    zc = z - z.mean(dim=0, keepdim=True)
    c = batch_cov(z)
    evals, evecs = torch.linalg.eigh(c)
    inv_sqrt = (evals.clamp_min(eps)).rsqrt()
    w = evecs @ torch.diag(inv_sqrt) @ evecs.T
    return zc @ w


def cholesky_whiten(z: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    zc = z - z.mean(dim=0, keepdim=True)
    c = batch_cov(z)
    d = c.shape[0]
    l = torch.linalg.cholesky(c + eps * torch.eye(d, device=c.device, dtype=c.dtype))
    return torch.linalg.solve_triangular(l, zc.T, upper=False).T


def ns_invsqrt_whiten(z: torch.Tensor, steps: int = 5, eps: float = 1e-6) -> torch.Tensor:
    zc = z - z.mean(dim=0, keepdim=True)
    w = ns_invsqrt_whiten_matrix(batch_cov(z), steps=steps, eps=eps)
    return zc @ w


def ns_muon_on_zt_project(z: torch.Tensor, steps: int = 5) -> torch.Tensor:
    """Architecture-style actuator: project through Muon-NS(Z^T) subspace."""
    zc = z - z.mean(dim=0, keepdim=True)
    u = newton_schulz5_zeroth(zc.T, steps=steps)  # (d, B)
    p = u @ u.T
    return zc @ p


def ns_muon_on_error(z: torch.Tensor, steps: int = 5, target: float = 1.0) -> torch.Tensor:
    """Apply Muon-NS to error matrix E = target*I - C (symmetric)."""
    zc = z - z.mean(dim=0, keepdim=True)
    d = zc.shape[1]
    c = batch_cov(z)
    e = target * torch.eye(d, device=z.device, dtype=z.dtype) - c
    e = 0.5 * (e + e.T)
    g = newton_schulz5_zeroth(e, steps=steps)
    return zc + zc @ g


def ns_muon_spread(z: torch.Tensor, steps: int = 5, gain: float = 1.0) -> torch.Tensor:
    """Spread deviation from batch mean using NS geometry on centered batch."""
    zc = z - z.mean(dim=0, keepdim=True)
    u = newton_schulz5_zeroth(zc.T, steps=steps)
    delta = (u.T @ zc.T).T  # (B, B) mixing — not whitening; test geometry spread
    return z.mean(dim=0, keepdim=True) + gain * delta


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic batch generators (controlled spectrum)
# ─────────────────────────────────────────────────────────────────────────────

def make_batch(
    scenario: str,
    batch: int,
    dim: int,
    device: torch.device,
    seed: int = 0,
) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    if scenario == "full_rank":
        z = torch.randn(batch, dim, generator=g, device=device)
    elif scenario == "anisotropic":
        z = torch.randn(batch, dim, generator=g, device=device)
        scales = torch.logspace(2, -2, dim, device=device)
        z = z * scales
    elif scenario == "exp_decay":
        z = torch.randn(batch, dim, generator=g, device=device)
        scales = torch.exp(-torch.arange(dim, device=device, dtype=torch.float32) * 0.35)
        z = z * scales
    elif scenario == "near_collapse":
        z = torch.randn(batch, dim, generator=g, device=device)
        z = z @ torch.diag(torch.tensor([1.0, 0.05, 0.02] + [0.01] * (dim - 3), device=device))
    elif scenario == "rank1":
        v = torch.randn(dim, generator=g, device=device)
        v = v / v.norm()
        z = torch.randn(batch, 1, generator=g, device=device) * v
    elif scenario == "isotropic":
        z = torch.randn(batch, dim, generator=g, device=device)
        z = z / z.std(dim=0, keepdim=True).clamp_min(1e-6)
    else:
        raise ValueError(f"unknown scenario: {scenario}")
    return z


# ─────────────────────────────────────────────────────────────────────────────
# NS singular-value behaviour (what NS actually does vs whitening)
# ─────────────────────────────────────────────────────────────────────────────

def analyze_ns_vs_svd_singular_values(dim: int = 32, device: torch.device | None = None) -> None:
    device = device or torch.device("cpu")
    print("\n" + "=" * 72)
    print("NS degree-5 vs true SVD: singular value maps on random matrix")
    print("=" * 72)
    g = torch.randn(dim, dim, device=device)
    u, s, vh = torch.linalg.svd(g, full_matrices=False)
    ns_g = newton_schulz5_zeroth(g, steps=5)
    _, s_ns, _ = torch.linalg.svd(ns_g, full_matrices=False)
    print(f"  Input  s: min={s.min():.4f}  max={s.max():.4f}  std/mean={s.std()/s.mean():.4f}")
    print(f"  NS out s: min={s_ns.min():.4f}  max={s_ns.max():.4f}  std/mean={s_ns.std()/s_ns.mean():.4f}")
    print("  => Muon NS balances singular values toward ~1, but output is UV'-like,")
    print("     NOT a map that makes Cov(Z) isotropic.")


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark runner
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MethodResult:
    name: str
    metrics: dict[str, float]
    ms: float
    ok: bool
    note: str = ""


def run_method(
    name: str,
    fn: Callable[[torch.Tensor], torch.Tensor],
    z: torch.Tensor,
) -> MethodResult:
    t0 = time.perf_counter()
    ok, note = True, ""
    try:
        z_out = fn(z)
        if not torch.isfinite(z_out).all():
            ok = False
            note = "non-finite output"
            z_out = z
    except Exception as exc:  # noqa: BLE001 — diagnostic script
        ok = False
        note = str(exc)[:80]
        z_out = z
    ms = (time.perf_counter() - t0) * 1000.0
    return MethodResult(name=name, metrics=isotropy_metrics(z_out), ms=ms, ok=ok, note=note)


def print_results(scenario: str, results: list[MethodResult], dim: int) -> None:
    print(f"\n{'─' * 72}")
    print(f"Scenario: {scenario}  (target: Cov -> I,  dim={dim})")
    print(f"{'─' * 72}")
    header = (
        f"{'Method':<28} {'fro_dev':>8} {'spec_std':>9} {'aniso':>10} "
        f"{'erank':>7} {'min_λ':>9} {'ms':>7}  status"
    )
    print(header)
    print("-" * len(header))
    for r in results:
        m = r.metrics
        status = "OK" if r.ok else f"FAIL ({r.note})"
        print(
            f"{r.name:<28} {m['fro_dev']:8.4f} {m['spec_std']:9.4f} "
            f"{m['aniso_ratio']:10.2f} {m['erank']:7.2f} {m['min_eval']:9.2e} "
            f"{r.ms:7.2f}  {status}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Isotropy transform benchmark")
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--ns-steps", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    dim, batch = args.dim, args.batch
    eps, ns_steps = args.eps, args.ns_steps

    scenarios = ["full_rank", "anisotropic", "exp_decay", "near_collapse", "rank1", "isotropic"]

    methods: list[tuple[str, Callable[[torch.Tensor], torch.Tensor]]] = [
        ("identity (no transform)", lambda z: z),
        ("SVD whiten", lambda z: svd_whiten(z, eps=eps)),
        ("Cholesky whiten", lambda z: cholesky_whiten(z, eps=eps)),
        ("NS inv-sqrt whiten (5 it)", lambda z: ns_invsqrt_whiten(z, steps=ns_steps, eps=eps)),
        ("NS Muon on Z^T project", lambda z: ns_muon_on_zt_project(z, steps=ns_steps)),
        ("NS Muon on E=I-C", lambda z: ns_muon_on_error(z, steps=ns_steps)),
    ]

    print("=" * 72)
    print("Isotropy benchmark: SVD / Cholesky / NS degree-5 variants")
    print(f"  dim={dim}  batch={batch}  eps={eps}  ns_steps={ns_steps}  device={device}")
    print("  Metrics: fro_dev=||C-I||/sqrt(d), spec_std=std(λ)/mean(λ), aniso=max/min(λ)")
    print("  Good isotropy: fro_dev~0, spec_std~0, aniso~1, erank~dim")
    print("=" * 72)

    # Input stats
    print("\nInput batch statistics (before transform):")
    for sc in scenarios:
        z = make_batch(sc, batch, dim, device, seed=args.seed)
        m = isotropy_metrics(z)
        print(
            f"  {sc:16s}  fro={m['fro_dev']:.3f}  aniso={m['aniso_ratio']:.1f}  "
            f"erank={m['erank']:.1f}  min_λ={m['min_eval']:.2e}"
        )

    for sc in scenarios:
        z = make_batch(sc, batch, dim, device, seed=args.seed)
        results = [run_method(name, fn, z) for name, fn in methods]
        print_results(sc, results, dim)

    analyze_ns_vs_svd_singular_values(dim=dim, device=device)

    print("\n" + "=" * 72)
    print("Summary")
    print("=" * 72)
    print("  • SVD / Cholesky whitening: designed to make Cov(Z_out) = I (isotropic).")
    print("  • NS inv-sqrt (5 iter): same goal via Newton-Schulz on C^{-1/2}; approximate.")
    print("  • NS Muon degree-5 (zeroth power): orthogonalizes update geometry; does NOT")
    print("    guarantee Cov -> I. Expect higher fro_dev / aniso vs whitening methods.")
    print("  • At rank-1 collapse, all methods struggle if input has zero variance;")
    print("    Cholesky/SVD rely on eps; Muon NS relies on Frobenius norm clamp.")
    print("=" * 72)


if __name__ == "__main__":
    main()
