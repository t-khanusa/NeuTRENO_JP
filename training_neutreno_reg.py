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

def spectrum(x: torch.Tensor, prefix: str, eps: float = 1e-8) -> dict[str, float]:
    """Effective rank / log det / min eigenvalue of the batch covariance."""
    z = x.detach().float()
    z = z - z.mean(0, keepdim=True)
    c = (z.T @ z) / max(z.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(0.5 * (c + c.T)).clamp_min(eps)
    p = ev / ev.sum()
    return {
        f"geom/erank_{prefix}": (ev.sum().square() / ev.square().sum()).item(),
        f"geom/entropy_rank_{prefix}": (-(p * p.log()).sum()).exp().item(),
        f"geom/logdet_{prefix}": ev.log().sum().item(),
        f"geom/var_{prefix}": ev.mean().item(),
    }


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

    wandb.init(project=cfg.get("wandb_project", "LeJEPA"), name=tag,
               group=reg, config=OmegaConf.to_container(cfg, resolve=True))
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
    if backbone_kind == "neutreno":
        net = NeuTRENOEncoder(
            img_size=img_size, patch_size=cfg.get("patch_size", 8),
            embed_dim=cfg.get("backbone_dim", 384), depth=cfg.get("depth", 12),
            num_heads=cfg.get("num_heads", 6), mlp_ratio=cfg.get("mlp_ratio", 4.0),
            out_dim=embed_dim, proj_dim=cfg.proj_dim,
            drop_path_rate=cfg.get("drop_path_rate", 0.1), lambda_p=lambda_p,
        ).to(device)
    elif backbone_kind == "timm":
        net = ViTEncoder(
            model_name=cfg.get("model_name", "vit_small_patch8_224"), img_size=img_size,
            embed_dim=embed_dim, proj_dim=cfg.proj_dim,
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
    ckpt_dir = os.path.join("checkpoints", tag)
    os.makedirs(ckpt_dir, exist_ok=True)

    # ── Training ─────────────────────────────────────────────────────────
    best_acc, step = 0.0, 0
    for epoch in range(cfg.epochs):
        net.train()
        probe.train()
        for i, (vs, y) in enumerate(tqdm.tqdm(train_loader, total=steps_per_epoch)):
            if i >= steps_per_epoch:
                break
            vs = vs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with autocast(device, dtype=torch.bfloat16):
                emb, proj = net(vs)
                yhat = probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y.repeat_interleave(V))

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
                log.update(spectrum(emb, "emb"))
                log.update(spectrum(proj.flatten(0, 1), "proj"))
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

        state = {"epoch": epoch, "net": net.state_dict(), "probe": probe.state_dict(),
                 "best_acc": best_acc, "cfg": OmegaConf.to_container(cfg, resolve=True)}
        if acc == best_acc:
            torch.save(state, os.path.join(ckpt_dir, "best.pt"))
        if (epoch + 1) % cfg.get("save_every", 100) == 0 or epoch + 1 == cfg.epochs:
            torch.save(state, os.path.join(ckpt_dir, "last.pt"))

    wandb.summary["final_acc"] = acc
    wandb.finish()
    print(f"\n[{tag}] Training complete. Best accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
