"""
NeuTRENO + LeJEPA Training (multi-mode).

Modes (cfg.mode):
  neutreno_lejepa  → NeuTRENO ViT + invariance + SIGReg
  only_neutreno    → NeuTRENO ViT + invariance only          (no SIGReg)
  only_pred        → timm ViT + invariance only              (no NeuTRENO, no SIGReg)
  lejepa           → timm ViT + invariance + SIGReg          (original LeJEPA)

NeuTRENO (Nguyen et al., NeurIPS 2024):
    layer 0:   keep V0
    layer ℓ>0: Attn(x) + λ_p (V0 − V(ℓ))

Usage:
    python training_neutreno_lejepa.py --config-path configs --config-name neutreno_lejepa
    python training_neutreno_lejepa.py mode=only_neutreno
    python training_neutreno_lejepa.py mode=only_pred
    python training_neutreno_lejepa.py mode=lejepa
    python training_neutreno_lejepa.py mode=neutreno_lejepa lambda_p=0.6
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

import wandb
import hydra
import tqdm
from omegaconf import DictConfig, OmegaConf

from training_lejepa import SIGReg, ViTEncoder, MultiCropDataset
from training_neutreno import NeuTRENOEncoder

MODES = ("neutreno_lejepa", "only_neutreno", "only_pred", "lejepa")


def resolve_mode(cfg: DictConfig) -> tuple[str, bool, bool]:
    """Return (mode, use_neutreno, use_sigreg)."""
    mode = str(cfg.get("mode", "neutreno_lejepa"))
    # Backward compat: old use_neutreno flag if mode absent from older overrides
    if "mode" not in cfg and "use_neutreno" in cfg:
        mode = "neutreno_lejepa" if cfg.use_neutreno else "lejepa"
    if mode not in MODES:
        raise ValueError(f"Unknown mode={mode!r}. Choose from {MODES}")
    use_neutreno = mode in ("neutreno_lejepa", "only_neutreno")
    use_sigreg = mode in ("neutreno_lejepa", "lejepa")
    return mode, use_neutreno, use_sigreg


@hydra.main(version_base=None, config_path="configs", config_name="neutreno_lejepa")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    mode, use_neutreno, use_sigreg = resolve_mode(cfg)
    lambda_p = float(cfg.get("lambda_p", 0.6)) if use_neutreno else 0.0

    tag = mode if mode != "neutreno_lejepa" else f"neutreno_lejepa_lp{lambda_p}"
    print(f"Mode: {mode}  |  NeuTRENO={use_neutreno}  SIGReg={use_sigreg}  lambda_p={lambda_p}")
    wandb.init(
        project="LeJEPA",
        name=tag,
        config=OmegaConf.to_container(cfg, resolve=True),
    )
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

    # ── Model ────────────────────────────────────────────────────────────
    embed_dim = cfg.get("embed_dim", 512)
    proj_dim = cfg.proj_dim

    if use_neutreno:
        print(f"Backbone: NeuTRENO ViT (lambda_p={lambda_p})")
        net = NeuTRENOEncoder(
            img_size=img_size,
            patch_size=cfg.get("patch_size", 8),
            embed_dim=cfg.get("backbone_dim", 384),
            depth=cfg.get("depth", 12),
            num_heads=cfg.get("num_heads", 6),
            mlp_ratio=cfg.get("mlp_ratio", 4.0),
            out_dim=embed_dim,
            proj_dim=proj_dim,
            drop_path_rate=cfg.get("drop_path_rate", 0.1),
            lambda_p=lambda_p,
        ).to(device)
    else:
        print(f"Backbone: timm ViT ({cfg.get('model_name', 'vit_small_patch8_224')})")
        net = ViTEncoder(
            model_name=cfg.get("model_name", "vit_small_patch8_224"),
            img_size=img_size,
            embed_dim=embed_dim,
            proj_dim=proj_dim,
            drop_path_rate=cfg.get("drop_path_rate", 0.1),
            pretrained=cfg.get("pretrained", False),
        ).to(device)

    probe = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)).to(device)
    sigreg = SIGReg().to(device) if use_sigreg else None

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])

    warmup_steps = len(train_loader)
    total_steps = len(train_loader) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=(device == "cuda"))

    # ── Training ─────────────────────────────────────────────────────────
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        net.train()
        probe.train()
        for vs, y in tqdm.tqdm(train_loader, total=len(train_loader)):
            with autocast(device, dtype=torch.bfloat16):
                vs = vs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                emb, proj = net(vs)

                batch_var = emb.var(dim=0).mean()

                # Prediction / invariance loss (always on)
                inv_loss = (proj.mean(0) - proj).square().mean()

                if use_sigreg:
                    sigreg_loss = sigreg(proj)
                    repr_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)
                else:
                    # only_neutreno / only_pred: invariance only
                    sigreg_loss = inv_loss.new_zeros(())
                    repr_loss = inv_loss

                # Online linear probe
                y_rep = y.repeat_interleave(V)
                yhat = probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y_rep)

                loss = repr_loss + probe_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            wandb.log({
                "train/probe": probe_loss.item(),
                "train/repr": repr_loss.item(),
                "train/sigreg": sigreg_loss.item(),
                "train/inv": inv_loss.item(),
                "train/batch_var": batch_var.item(),
                "train/use_neutreno": float(use_neutreno),
                "train/use_sigreg": float(use_sigreg),
                "train/lambda_p": lambda_p,
            })

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
        print(f"Epoch {epoch+1}: acc={acc*100:.2f}% (best={best_acc*100:.2f}%)")

        if (epoch + 1) % cfg.get("save_every", 100) == 0 or acc == best_acc:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch, "net": net.state_dict(), "probe": probe.state_dict(),
                "optimizer": opt.state_dict(), "best_acc": best_acc,
                "mode": mode, "use_neutreno": use_neutreno, "use_sigreg": use_sigreg,
                "lambda_p": lambda_p,
            }, f"checkpoints/{tag}_epoch{epoch+1}_acc{acc*100:.1f}.pt")

    wandb.finish()
    print(f"\nTraining complete. Best accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
