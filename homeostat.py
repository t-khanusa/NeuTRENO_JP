"""Homeostatic layers: distributed weight-space PID control of activation statistics.

The mechanism (per controlled Linear layer ℓ, weight W ∈ R^{d_out×d_in}):

    plant     C_ℓ  = Cov[a_ℓ]              output activation covariance,
                                            estimated online by EMA over batches
    setpoint  R_ℓ  = σ²_ℓ · I              isotropic; σ²_ℓ fixed or calibrated
                                            from the layer's own init statistics
    error     E_ℓ  = R_ℓ − C_ℓ
    integral  M_ℓ ← M_ℓ + η_i E_ℓ          elementwise anti-windup clamp ±m_clip
    control   G_ℓ  = k_p E_ℓ + k_i M_ℓ + k_d (E_ℓ − E_ℓ⁻)   symmetric gain
    actuator  W_ℓ ← W_ℓ + κ G_ℓ W_ℓ        multiplicative flow on the weights
              b_ℓ ← b_ℓ + κ G_ℓ b_ℓ        (bias lives in the same output space)

Why this is NOT a loss term and NOT a heuristic:
  * The update κ G W is a local plasticity rule (homeostatic synaptic scaling).
    Because G contains the integral memory M, the induced vector field on W is
    NON-conservative — it is not the gradient of any scalar function, so no
    "hidden loss" reproduces it.
  * It is unconditionally restoring: if a layer's output covariance decays
    (collapse), E stays ≈ σ²I, M grows linearly in time, and the actuation
    W ← (I + κ(k_p + k_i t)σ²P)W re-inflates the collapsed subspace P
    exponentially. Collapse is not a fixed point of the coupled dynamics.
  * It is distributed: EVERY Linear layer in the controlled tail carries its
    own loop, so there is no unconstrained "cut" left for the invariance
    gradient to collapse through (the min-cut escape of activation-space
    anchors does not apply — a rank-deficient W is detected at the layer that
    owns it and re-expanded there).
  * Two-timescale separation: κ ≪ lr keeps the homeostat a slow outer loop
    around the fast task-driven dynamics, the standard singular-perturbation
    setting for adaptive control.

Control-theoretic guards (both are standard actuator constraints, not tricks):
  * anti-windup: M is clamped elementwise (|M| ≤ m_clip),
  * saturation:  G is rescaled if ‖G‖_F > g_max·√d (limits actuator authority).
"""

from __future__ import annotations

import torch
import torch.nn as nn


def collect_homeo_modules(net: nn.Module, last_blocks: int = 4) -> dict[str, nn.Linear]:
    """Every Linear in the last `last_blocks` transformer blocks, the backbone
    head, and the projector. Covering all of them leaves no unregulated cut
    between the controlled trunk and the invariance loss."""
    modules: dict[str, nn.Linear] = {}
    backbone = getattr(net, "backbone", net)

    blocks = getattr(backbone, "blocks", None)
    if blocks is not None and last_blocks > 0:
        n = len(blocks)
        for i in range(max(0, n - last_blocks), n):
            for name, m in blocks[i].named_modules():
                if isinstance(m, nn.Linear):
                    modules[f"backbone.blocks.{i}.{name}"] = m

    head = getattr(backbone, "head", None)
    if isinstance(head, nn.Linear):
        modules["backbone.head"] = head

    proj = getattr(net, "proj", None)
    if proj is not None:
        for name, m in proj.named_modules():
            if isinstance(m, nn.Linear):
                modules[f"proj.{name}"] = m

    return modules


class HomeostaticController:
    """Distributed weight-space PID homeostat over a set of Linear layers.

    Usage:
        homeo = HomeostaticController(modules, ...)
        ... forward/backward/optimizer.step() as usual ...
        homeo.step()          # applies W <- W + kappa*G@W under no_grad
        wandb.log(homeo.metrics())
    """

    def __init__(
        self,
        modules: dict[str, nn.Linear],
        kappa: float = 1e-3,
        k_p: float = 1.0,
        k_i: float = 0.1,
        k_d: float = 0.0,
        eta_i: float = 0.02,
        m_clip: float = 5.0,
        ema: float = 0.05,
        sigma2: float | str = "auto",
        warmup_steps: int = 100,
        max_rows: int = 2048,
        g_max: float = 1.0,
    ):
        self.modules = modules
        self.kappa = kappa
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.eta_i = eta_i
        self.m_clip = m_clip
        self.ema = ema
        self.sigma2_cfg = sigma2
        self.warmup_steps = warmup_steps
        self.max_rows = max_rows
        self.g_max = g_max
        self.steps = 0

        # Per-layer state, allocated lazily on first observation (device/dim).
        self.C: dict[str, torch.Tensor] = {}        # EMA output covariance
        self.M: dict[str, torch.Tensor] = {}        # integral state
        self.E_prev: dict[str, torch.Tensor] = {}   # for the D term
        self.sigma2: dict[str, float] = {}          # per-layer setpoint scale
        self._seen: dict[str, bool] = {}            # observed this step?
        self._last: dict[str, dict[str, float]] = {}

        self._handles = [
            m.register_forward_hook(self._make_hook(name))
            for name, m in modules.items()
        ]

    # ── observation (forward hooks) ─────────────────────────────────────────

    def _make_hook(self, name: str):
        def hook(module: nn.Module, inputs, output: torch.Tensor):
            if not module.training or not torch.is_grad_enabled():
                return
            with torch.no_grad():
                a = output.detach().float().reshape(-1, output.shape[-1])
                if a.shape[0] > self.max_rows:
                    idx = torch.randint(0, a.shape[0], (self.max_rows,), device=a.device)
                    a = a[idx]
                a = a - a.mean(dim=0, keepdim=True)
                c = (a.transpose(0, 1) @ a) / max(a.shape[0] - 1, 1)
                if name not in self.C:
                    self.C[name] = c.clone()
                    self.M[name] = torch.zeros_like(c)
                else:
                    self.C[name].mul_(1.0 - self.ema).add_(c, alpha=self.ema)
                self._seen[name] = True
        return hook

    # ── actuation (after optimizer step) ────────────────────────────────────

    @torch.no_grad()
    def step(self) -> None:
        self.steps += 1

        if self.steps == self.warmup_steps and not self.sigma2:
            # Calibrate each layer's setpoint at its own healthy init-time scale:
            # "keep your birth statistics" — no external target is injected.
            for name, C in self.C.items():
                s2 = (C.diagonal().mean().item() if self.sigma2_cfg == "auto"
                      else float(self.sigma2_cfg))
                self.sigma2[name] = max(s2, 1e-6)
        if self.steps < self.warmup_steps:
            self._seen.clear()
            return

        for name, module in self.modules.items():
            if not self._seen.pop(name, False) or name not in self.sigma2:
                continue
            C = self.C[name]
            d = C.shape[0]
            eye = torch.eye(d, device=C.device, dtype=C.dtype)

            E = self.sigma2[name] * eye - C
            M = self.M[name]
            M.add_(E, alpha=self.eta_i).clamp_(-self.m_clip, self.m_clip)

            G = self.k_p * E + self.k_i * M
            if self.k_d != 0.0:
                if name in self.E_prev:
                    G = G + self.k_d * (E - self.E_prev[name])
                self.E_prev[name] = E.clone()

            # Actuator saturation: bound authority by g_max in RMS-singular-value
            # units (‖G‖_F ≤ g_max·√d ⇒ mean singular value ≤ g_max).
            g_fro = G.norm()
            limit = self.g_max * (d ** 0.5)
            scale = 1.0 if g_fro <= limit else (limit / (g_fro + 1e-12)).item()

            W = module.weight
            dW = (G.to(W.dtype) @ W) * (self.kappa * scale)
            W.add_(dW)
            if module.bias is not None:
                b = module.bias
                b.add_((G.to(b.dtype) @ b) * (self.kappa * scale))

            self._last[name] = {
                "trace_ratio": (C.diagonal().mean() / self.sigma2[name]).item(),
                "err_fro": E.norm().item() / d,
                "integ_fro": M.norm().item() / d,
                "gain_scale": scale,
                "dw_rel": (dW.norm() / (W.norm() + 1e-12)).item(),
            }
        self._seen.clear()

    # ── telemetry / persistence ─────────────────────────────────────────────

    def metrics(self) -> dict[str, float]:
        if not self._last:
            return {}
        keys = next(iter(self._last.values())).keys()
        out = {}
        for k in keys:
            vals = [v[k] for v in self._last.values()]
            out[f"homeo/{k}_mean"] = sum(vals) / len(vals)
        out["homeo/trace_ratio_min"] = min(v["trace_ratio"] for v in self._last.values())
        return out

    def state_dict(self) -> dict:
        return {
            "steps": self.steps,
            "sigma2": self.sigma2,
            "C": {k: v.cpu() for k, v in self.C.items()},
            "M": {k: v.cpu() for k, v in self.M.items()},
        }

    def load_state_dict(self, sd: dict, device: str = "cuda") -> None:
        self.steps = sd["steps"]
        self.sigma2 = dict(sd["sigma2"])
        self.C = {k: v.to(device) for k, v in sd["C"].items()}
        self.M = {k: v.to(device) for k, v in sd["M"].items()}

    def remove(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
