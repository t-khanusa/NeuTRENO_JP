"""Screened-diffusion JEPA: the sample-axis mirror of NeuTRENO.

Theory. The invariance-loss gradient flow on representations is heat diffusion
on the augmentation graph:  ż = −L_aug z  → global constant (collapse), the
exact sample-axis analogue of token over-smoothing along depth. NeuTRENO's cure
on the token axis is a fidelity term pulling values toward the flow's initial
state V0. Mirrored to the sample axis, the initial state of TRAINING is the
random-init encoder z0(x) = f_{θ0}(x) — frozen, exogenous, full-rank (random
features), so it cannot co-collapse. The tethered flow

    ż = −(L_aug + λ_s I) z + λ_s z0        (screened Poisson equation)

has the unique non-constant equilibrium  z* = λ_s (L_aug + λ_s I)^{-1} z0 :
random features low-pass filtered on the augmentation graph. Semantics come
from the GRAPH FILTER (augmentation-invariant eigenmodes survive, HaoChen et
al.), not from the anchor — the anchor only pins the constant mode, exactly
like V0 in NeuTRENO. Prediction: probe accuracy vs λ_s is an inverted U.

Loss = multi-view invariance + λ_s ||emb − emb0||²  (+ detached online probe).
NO SIGReg/VICReg, NO whitening, NO EMA teacher, NO stop-grad between views.

Cascade option: backbone can be NeuTRENO (lambda_p > 0) → P-control on BOTH
axes (token graph along depth + sample graph along training time).

Usage:
    python training_tether.py --config-path configs --config-name tether
    python training_tether.py --config-path configs --config-name tether tether_lambda=0.1 lambda_p=0.6
    bash grid_tether.sh
"""

import os
import copy

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

from training_pid_sample import MultiCropDataset
from training_neutreno import NeuTRENOEncoder, proj_geometry


@hydra.main(version_base=None, config_path="configs", config_name="tether")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    lambda_p = cfg.get("lambda_p", 0.0)
    tether_lambda = cfg.get("tether_lambda", 0.1)
    run_name = f"tether_ls{tether_lambda}_lp{lambda_p}"
    if tether_lambda == 0:
        run_name = f"tether_off_lp{lambda_p}_baseline"
    wandb.init(project="PID-Tether", name=run_name,
               config=OmegaConf.to_container(cfg, resolve=True))
    torch.manual_seed(cfg.get("seed", 0))
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Dataset (identical to the other pipelines) ─────────────────────────
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

    train_loader = DataLoader(train_ds, batch_size=cfg.bs, shuffle=True,
                              drop_last=True, num_workers=cfg.get("num_workers", 8),
                              pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256,
                             num_workers=cfg.get("num_workers", 8), pin_memory=True)

    # ── Model (NeuTRENO backbone; lambda_p=0 → plain ViT dynamics) ─────────
    embed_dim = cfg.get("embed_dim", 512)
    proj_dim = cfg.proj_dim
    print(f"Backbone: NeuTRENO ViT (lambda_p={lambda_p}), tether λ_s={tether_lambda}")
    net = NeuTRENOEncoder(
        img_size=img_size, patch_size=cfg.get("patch_size", 8),
        embed_dim=cfg.get("backbone_dim", 384), depth=cfg.get("depth", 12),
        num_heads=cfg.get("num_heads", 6), mlp_ratio=cfg.get("mlp_ratio", 4.0),
        out_dim=embed_dim, proj_dim=proj_dim,
        drop_path_rate=cfg.get("drop_path_rate", 0.1),
        lambda_p=lambda_p,
    ).to(device)

    # ── The anchor: frozen copy of the backbone AT RANDOM INIT ─────────────
    # z0(x) is exogenous (never trained) and full-rank on data (random
    # features), so the tethered training flow cannot reach the constant mode.
    anchor = None
    if tether_lambda > 0:
        anchor = copy.deepcopy(net.backbone).to(device)
        anchor.eval()
        for p in anchor.parameters():
            p.requires_grad_(False)

    probe = nn.Sequential(nn.LayerNorm(embed_dim),
                          nn.Linear(embed_dim, num_classes)).to(device)

    # ── Optimizer (identical to the other pipelines) ───────────────────────
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])

    warmup_steps = len(train_loader)
    total_steps = len(train_loader) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=max(total_steps - warmup_steps, 1), eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=(device == "cuda"))

    # ── Training ───────────────────────────────────────────────────────────
    best_acc = 0.0
    for epoch in range(cfg.epochs):
        net.train()
        probe.train()
        for vs, y in tqdm.tqdm(train_loader, total=len(train_loader)):
            vs = vs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device, dtype=torch.bfloat16):
                emb, proj = net(vs)
                if anchor is not None:
                    with torch.no_grad():
                        emb0 = anchor(vs.flatten(0, 1))

            proj = proj.float()
            Vv, Nn, P = proj.shape
            proj_flat = proj.reshape(Vv * Nn, P)

            # Diffusion term: multi-view invariance (the ONLY semantic loss).
            inv_loss = (proj.mean(0) - proj).square().mean()

            # Fidelity term: screened-diffusion tether to the init-time state.
            if anchor is not None:
                tether_loss = (emb.float() - emb0.float()).square().mean()
            else:
                tether_loss = torch.zeros((), device=device)

            y_rep = y.repeat_interleave(V)
            yhat = probe(emb.detach().float())
            probe_loss = F.cross_entropy(yhat, y_rep)

            loss = inv_loss + tether_lambda * tether_loss + probe_loss

            batch_var = emb.detach().float().var(dim=0).mean()
            proj_var = proj_flat.detach().var(dim=0).mean()

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()

            log = {
                "train/inv_loss": inv_loss.item(),
                "train/tether_loss": tether_loss.item(),
                "train/probe_loss": probe_loss.item(),
                "train/total_loss": loss.item(),
                "train/batch_var": batch_var.item(),
                "train/proj_var": proj_var.item(),
            }
            log.update(proj_geometry(proj_flat))
            wandb.log(log)

        # ── Evaluation ─────────────────────────────────────────────────────
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
            torch.save(
                {"epoch": epoch, "net": net.state_dict(), "probe": probe.state_dict(),
                 "optimizer": opt.state_dict(), "best_acc": best_acc},
                f"checkpoints/tether_ls{tether_lambda}_lp{lambda_p}_epoch{epoch+1}_acc{acc*100:.1f}.pt")

    wandb.finish()
    print(f"\nTraining complete. Best accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
