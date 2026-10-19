"""
NeuTRENO × anti-collapse regularizer.

Crosses the backbone axis (token-level P-control, NeuTRENO) with the loss axis
(sample-level anti-collapse regularizer) so the effect of NeuTRENO can be read
off for each regularizer separately:

    backbone=neutreno lambda_p=0.6   NeuTRENO ViT-S/8
    backbone=neutreno lambda_p=0.0   the SAME ViT-S/8 without the value residual
                                     (controls for implementation/init differences
                                     against the timm ViT)
    backbone=timm                    timm vit_small_patch8_224 (the original LeJEPA run)

    reg ∈ anticollapse.REGS          sigreg | visreg | lambdajepa | sacreg | vicreg
                                     | barlow | uniformity | none

Usage:
    python training_neutreno_reg.py reg=visreg lambda_p=0.6
    python training_neutreno_reg.py reg=lambdajepa lambda_p=0.0
    python training_neutreno_reg.py reg=sigreg backbone=timm
    bash grid_neutreno_reg.sh
"""

import os
import random

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import wandb
import hydra
import tqdm
from omegaconf import DictConfig, OmegaConf

from training_lejepa import ViTEncoder, MultiCropDataset
from training_neutreno import NeuTRENOEncoder, NeuTRENOViT, NeuTRENOAttention
from anticollapse import AntiCollapse


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry
# ─────────────────────────────────────────────────────────────────────────────

def spectrum(x: torch.Tensor, prefix: str, eps: float = 1e-8,
             collapse_tol: float = 1e-8) -> dict[str, float]:
    """Spectrum of the covariance of x (n, d).

    participation_rank = (Σλ)² / Σλ²      entropy_rank = exp(H(λ / Σλ))
    Neither is RankMe (that is the entropy rank of the singular values of the
    uncentred embedding matrix). Both use λ ≥ 0 directly — only round-off
    negatives are clipped. If the total variance is ≈ 0 relative to the second
    moment (collapse to a point) the ranks are undefined; they are reported as 0
    and geom/collapsed_* = 1 instead of the d a floor on λ would give. eps enters
    log det only.
    """
    z = x.detach().double()
    d = z.shape[1]
    second_moment = z.square().sum(1).mean()
    z = z - z.mean(0, keepdim=True)
    c = (z.T @ z) / max(z.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(0.5 * (c + c.T)).clamp_min(0)
    tot = ev.sum()
    collapsed = bool(tot <= collapse_tol * second_moment.clamp_min(1e-30))
    if collapsed:
        pr = er = top = 0.0
    else:
        p = ev / tot
        pr = (tot.square() / ev.square().sum()).item()
        er = torch.exp(-torch.special.xlogy(p, p).sum()).item()
        top = p.max().item()
    return {
        f"geom/participation_rank_{prefix}": pr,
        f"geom/entropy_rank_{prefix}": er,
        f"geom/top_eig_frac_{prefix}": top,
        f"geom/logdet_{prefix}": torch.log(ev + eps).sum().item(),
        f"geom/var_{prefix}": (tot / d).item(),
        f"geom/collapsed_{prefix}": float(collapsed),
    }


def view_geometry(x: torch.Tensor, prefix: str) -> dict[str, float]:
    """Geometry of x (V, N, D) = views × images.

    The pooled image×view covariance mixes augmentation variance into the rank,
    so the spectrum is also taken on the per-image view centres (between-image),
    next to the within-image variance across views (per dim, unbiased over V).
    """
    x = x.detach().float()
    out = spectrum(x.flatten(0, 1), prefix)
    out.update(spectrum(x.mean(0), f"{prefix}_center"))
    within = x.var(0).mean().item()
    between = out[f"geom/var_{prefix}_center"]
    out[f"geom/within_var_{prefix}"] = within
    out[f"geom/within_over_between_{prefix}"] = within / between if between > 0 else float("inf")
    return out


@torch.no_grad()
def final_tokens(backbone, x):
    """Normed last-layer tokens (B, 1+L, C) of either backbone, without the head."""
    if isinstance(backbone, NeuTRENOViT):
        h = backbone.patch_embed(x)
        h = torch.cat((backbone.cls_token.expand(h.shape[0], -1, -1), h), dim=1)
        h = backbone.pos_drop(h + backbone.pos_embed)
        h, v0 = backbone.blocks[0](h)
        for blk in backbone.blocks[1:]:
            h = blk(h, v0=v0)
        return backbone.norm(h)
    return backbone.forward_features(x)


@torch.no_grad()
def token_smoothness(backbone, x) -> dict[str, float]:
    """Token-axis over-smoothing: mean pairwise cosine of patch tokens within an image."""
    t = F.normalize(final_tokens(backbone, x)[:, 1:].float(), dim=-1)
    L = t.shape[1]
    cos = (t @ t.transpose(1, 2)).sum((1, 2)).sub(L).div(L * (L - 1))
    return {"geom/token_cos": cos.mean().item()}


# ─────────────────────────────────────────────────────────────────────────────
# Model
# ─────────────────────────────────────────────────────────────────────────────

def build_net(cfg, device) -> nn.Module:
    """Encoder (backbone + projector) of a run; shared with posthoc_analysis.py."""
    backbone_kind = str(cfg.get("backbone", "neutreno"))
    lambda_p = float(cfg.get("lambda_p", 0.6)) if backbone_kind == "neutreno" else 0.0
    img_size = cfg.get("img_size", 128)
    embed_dim = cfg.get("embed_dim", 512)
    if backbone_kind == "neutreno":
        net = NeuTRENOEncoder(
            img_size=img_size, patch_size=cfg.get("patch_size", 8),
            embed_dim=cfg.get("backbone_dim", 384), depth=cfg.get("depth", 12),
            num_heads=cfg.get("num_heads", 6), mlp_ratio=cfg.get("mlp_ratio", 4.0),
            out_dim=embed_dim, proj_dim=cfg.get("proj_dim", 16),
            drop_path_rate=cfg.get("drop_path_rate", 0.1), lambda_p=lambda_p,
        ).to(device)
    elif backbone_kind == "timm":
        net = ViTEncoder(
            model_name=cfg.get("model_name", "vit_small_patch8_224"), img_size=img_size,
            embed_dim=embed_dim, proj_dim=cfg.get("proj_dim", 16),
            drop_path_rate=cfg.get("drop_path_rate", 0.1),
            pretrained=cfg.get("pretrained", False),
        ).to(device)
    else:
        raise ValueError(f"Unknown backbone={backbone_kind!r}")

    # Memory: bs=128 × 8 views does not fit a 24 GB GPU otherwise. Both are exact
    # (checkpointing replays the dropout/drop-path RNG; SDPA = softmax(qkᵀ/√d)v).
    if cfg.get("grad_ckpt", True):
        if backbone_kind == "neutreno":
            net.backbone.grad_checkpointing = True
        else:
            net.backbone.set_grad_checkpointing(True)
    if backbone_kind == "neutreno":
        NeuTRENOAttention.fused_attn = cfg.get("fused_attn", True)
    return net


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def run_dir_name(tag: str, cfg) -> str:
    """Unique per config: budget changes the cosine schedule, so it is part of the name."""
    return f"{tag}_{cfg.get('dataset', 'imagenette')}_s{cfg.get('seed', 0)}_ep{cfg.epochs}"


def rng_state() -> dict:
    return {"torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy": np.random.get_state(), "python": random.getstate()}


def set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    if st["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])


def atomic_save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="neutreno_reg")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    reg = str(cfg.reg)
    backbone_kind = str(cfg.get("backbone", "neutreno"))
    lambda_p = float(cfg.get("lambda_p", 0.6)) if backbone_kind == "neutreno" else 0.0
    # names follow the LeJEPA project (neutreno0.6_lejepa / lejepa-sigreg)
    bb_tag = f"neutreno{lambda_p}" if backbone_kind == "neutreno" else "timm"
    tag = cfg.get("run_name") or f"{bb_tag}_{reg}"
    print(f"reg={reg}  backbone={backbone_kind}  lambda_p={lambda_p}")

    seed = cfg.get("seed", 0)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Dataset ──────────────────────────────────────────────────────────
    V_global = cfg.get("V_global", 2)
    V_local = cfg.get("V_local", 6)
    V = V_global + V_local
    img_size = cfg.get("img_size", 128)
    dataset_name = cfg.get("dataset", "imagenette")
    num_classes = cfg.get("num_classes", 10)

    train_ds = MultiCropDataset("train", V_global=V_global, V_local=V_local,
                                img_size=img_size, dataset_name=dataset_name)
    test_ds = MultiCropDataset("validation", V_global=0, V_local=0,
                               img_size=img_size, dataset_name=dataset_name)
    test_ds.V = 1

    train_loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True, drop_last=True,
                              num_workers=cfg.get("num_workers", 8), pin_memory=True,
                              persistent_workers=True)
    test_loader = DataLoader(test_ds, batch_size=256, num_workers=cfg.get("num_workers", 8),
                             pin_memory=True)

    # ── Model ────────────────────────────────────────────────────────────
    embed_dim = cfg.get("embed_dim", 512)
    net = build_net(cfg, device)

    # raw CLS (input of the 384→512 head) for telemetry, image-major (N*V, C).
    # Cloned, and only on geometry steps: inp[0] is a view into the full token
    # tensor, which would otherwise stay alive into the next step.
    cls_cache = {"on": False}

    def grab_cls(module, inp, out):
        if cls_cache["on"]:
            cls_cache["cls"] = inp[0].detach().clone()

    net.backbone.head.register_forward_hook(grab_cls)

    probe = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)).to(device)
    anti = AntiCollapse(cfg, n_global=V_global).to(device)

    # ── Optimizer & Scheduler (identical to training_neutreno_lejepa.py) ─
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])

    limit = cfg.get("limit_batches")
    steps_per_epoch = min(len(train_loader), limit) if limit else len(train_loader)
    warmup_steps = steps_per_epoch
    total_steps = steps_per_epoch * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    geom_every = cfg.get("geom_every", 50)

    # ── Checkpoint dir & resume ──────────────────────────────────────────
    # resume=auto continues from <dir>/last.pt if it exists; resume=<path> from that
    # file; resume=never starts fresh and refuses to overwrite an existing run dir
    # unless overwrite=true. Resume restores net/probe/opt/sched/RNG/step, so the
    # LR schedule and the sampler continue; augmentation streams of the (new)
    # dataloader workers are re-seeded, so it is not bit-identical to no interruption.
    ckpt_dir = os.path.join(cfg.get("ckpt_root", "checkpoints"), run_dir_name(tag, cfg))
    last_path = os.path.join(ckpt_dir, "last.pt")
    resume = str(cfg.get("resume", "auto"))
    resume_path = (last_path if resume == "auto" and os.path.exists(last_path)
                   else resume if resume not in ("auto", "never") else None)
    if resume_path is None and os.path.exists(last_path) and not cfg.get("overwrite", False):
        raise FileExistsError(f"{ckpt_dir} holds a run; use resume=auto or overwrite=true")
    os.makedirs(os.path.join(ckpt_dir, "snapshots"), exist_ok=True)

    best_acc, step, start_epoch, wandb_id = 0.0, 0, 0, None
    if resume_path is not None:
        ck = torch.load(resume_path, map_location="cpu", weights_only=False)  # RNG states must stay on CPU
        net.load_state_dict(ck["net"])
        probe.load_state_dict(ck["probe"])
        opt.load_state_dict(ck["opt"])
        scheduler.load_state_dict(ck["sched"])
        set_rng_state(ck["rng"])
        best_acc, step, start_epoch = ck["best_acc"], ck["step"], ck["epoch"] + 1
        wandb_id = ck.get("wandb_id")
        print(f"[{tag}] resumed from {resume_path} at epoch {start_epoch} (step {step})")

    run = wandb.init(project=cfg.get("wandb_project", "LeJEPA"), name=tag, group=reg,
                     config={**OmegaConf.to_container(cfg, resolve=True), "ckpt_dir": ckpt_dir},
                     id=wandb_id, resume="allow" if wandb_id else None)
    snapshot_epochs = set(cfg.get("snapshot_epochs") or [])

    # ── Training ─────────────────────────────────────────────────────────
    acc = float("nan")
    for epoch in range(start_epoch, cfg.epochs):
        net.train()
        probe.train()
        for i, (vs, y) in enumerate(tqdm.tqdm(train_loader, total=steps_per_epoch)):
            if i >= steps_per_epoch:
                break
            vs = vs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            cls_cache["on"] = step % geom_every == 0
            with autocast(device, dtype=torch.bfloat16):
                emb, proj = net(vs)
                yhat = probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y.repeat_interleave(V))
            cls_cache["on"] = False

            # emb is image-major (N*V, E); the regularizers expect view-major (V, N, E)
            emb_v = emb.reshape(-1, V, emb.shape[-1]).transpose(0, 1)
            repr_loss, terms = anti(proj, emb_v)
            loss = repr_loss + probe_loss

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            scheduler.step()

            log = {"train/repr": repr_loss.item(), "train/probe": probe_loss.item(),
                   "train/batch_var": emb.detach().float().var(dim=0).mean().item(),
                   "train/lr": scheduler.get_last_lr()[0]}
            log.update({f"train/{k}": v.item() for k, v in terms.items()})
            if not torch.isfinite(repr_loss):
                raise FloatingPointError(f"non-finite repr_loss at step {step}: {log}")
            if step % geom_every == 0:
                cls = cls_cache.pop("cls")
                cls_v = cls.reshape(-1, V, cls.shape[-1]).transpose(0, 1)
                log.update(view_geometry(cls_v, "cls"))
                log.update(view_geometry(emb_v, "emb"))
                log.update(view_geometry(proj, "proj"))
                net.eval()
                log.update(token_smoothness(net.backbone, vs[:16, 0]))
                net.train()
            wandb.log(log)
            step += 1

        # ── Evaluation ───────────────────────────────────────────────────
        net.eval()
        probe.eval()
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
        print(f"[{tag}] Epoch {epoch+1}: acc={acc*100:.2f}% (best={best_acc*100:.2f}%) "
              f"peak_mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GB")

        # weights-only files for analysis; last.pt is the full resumable state
        weights = {"epoch": epoch, "net": net.state_dict(), "probe": probe.state_dict(),
                   "acc": acc, "best_acc": best_acc,
                   "cfg": OmegaConf.to_container(cfg, resolve=True)}
        if acc == best_acc:
            atomic_save(weights, os.path.join(ckpt_dir, "best.pt"))
        if epoch + 1 in snapshot_epochs:
            atomic_save(weights, os.path.join(ckpt_dir, "snapshots", f"ep{epoch + 1:04d}.pt"))
        if (epoch + 1) % cfg.get("save_every", 1) == 0 or epoch + 1 == cfg.epochs:
            atomic_save({**weights, "opt": opt.state_dict(), "sched": scheduler.state_dict(),
                         "rng": rng_state(), "step": step, "wandb_id": run.id}, last_path)

    if start_epoch < cfg.epochs:
        wandb.summary["final_acc"] = acc
    wandb.finish()
    print(f"\n[{tag}] Training complete. Best accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
