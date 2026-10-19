"""
PID-JEPA Architecture: sample-level PID + NS whitening actuator (forward pass).

Same PID shell as training_pid_sample.py (Steps 1–7), but the actuator uses
Newton–Schulz matrix functions instead of Cholesky:

  z̃ → z̃ @ NS_invsqrt(C + εI) @ NS_sqrt(S_eff)
  S_eff = R + (K_i·W + K_d·dE) / K_p
  z_out = μ + (1 − blend)·z̃ + blend·z_ctrl

Loss = invariance ONLY (+ probe for monitoring). No ctrl_loss / SIGReg / VICReg.

Usage:
    python training_pid_arch.py --config-path configs --config-name pid_arch
    python training_pid_arch.py +token_pid=false
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

from covariance_pid_actuator import CovariancePIDActuator
from training_pid_sample import (
    PIDformerEncoder,
    ViTEncoder,
    MultiCropDataset,
)


@hydra.main(version_base=None, config_path="configs", config_name="pid_arch")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    wandb.init(project="PID-Arch", config=OmegaConf.to_container(cfg, resolve=True))
    torch.manual_seed(cfg.get("seed", 0))
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    cov_act = CovariancePIDActuator(
        dim=proj_dim,
        sigma2=cfg.get("sigma2", 1.0),
        k_p=cfg.get("cov_k_p", 1.0),
        k_i=cfg.get("cov_k_i", 0.3),
        k_d=cfg.get("cov_k_d", 0.05),
        eta_i=cfg.get("cov_eta_i", 0.02),
        w_clip=cfg.get("cov_w_clip", 10.0),
        normalize=cfg.get("cov_normalize", True),
        one_sided=cfg.get("cov_one_sided", True),
        ns_steps=cfg.get("ns_steps", 5),
        cov_eps=cfg.get("cov_eps", 1e-4),
        ns_eps=cfg.get("ns_eps", 1e-7),
        ns_a=cfg.get("ns_a", 3.4445),
        ns_b=cfg.get("ns_b", -4.7750),
        ns_c=cfg.get("ns_c", 2.0315),
        ns_poly=cfg.get("ns_poly", "muon5"),
        actuator_blend=cfg.get("actuator_blend", None),
    ).to(device)

    probe = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)).to(device)

    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])

    warmup_steps = len(train_loader)
    total_steps = len(train_loader) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
    scheduler = SequentialLR(opt, schedulers=[s1, s2], milestones=[warmup_steps])

    scaler = GradScaler(enabled=(device == "cuda"))
    best_acc = 0.0

    for epoch in range(cfg.epochs):
        net.train()
        probe.train()
        cov_act.train()
        for vs, y in tqdm.tqdm(train_loader, total=len(train_loader)):
            vs = vs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device, dtype=torch.bfloat16):
                emb, proj = net(vs)

            proj = proj.float()
            vv, nn_batch, p = proj.shape
            proj_flat = proj.reshape(vv * nn_batch, p)

            # Architecture PID: NS whitening actuator in forward pass (no ctrl loss)
            proj_ctrl_flat, ctrl = cov_act(proj_flat)
            proj_ctrl = proj_ctrl_flat.reshape(vv, nn_batch, p)

            inv_loss = (proj_ctrl.mean(0) - proj_ctrl).square().mean()

            y_rep = y.repeat_interleave(V)
            yhat = probe(emb.detach().float())
            probe_loss = F.cross_entropy(yhat, y_rep)

            loss = inv_loss + probe_loss

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
            log.update(ctrl)
            wandb.log(log)

        net.eval()
        probe.eval()
        cov_act.eval()
        correct = 0
        with torch.inference_mode():
            for vs, y in test_loader:
                vs = vs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast(device, dtype=torch.bfloat16):
                    emb, proj = net(vs)
                proj_flat = proj.float().reshape(-1, proj_dim)
                proj_ctrl, _ = cov_act(proj_flat)
                correct += (probe(emb.float()).argmax(1) == y).sum().item()
        acc = correct / len(test_ds)
        best_acc = max(best_acc, acc)
        wandb.log({"test/acc": acc, "test/best_acc": best_acc, "test/epoch": epoch})
        print(f"Epoch {epoch+1}: acc={acc*100:.2f}% (best={best_acc*100:.2f}%)")

        if (epoch + 1) % cfg.get("save_every", 100) == 0 or acc == best_acc:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save({
                "epoch": epoch,
                "net": net.state_dict(),
                "probe": probe.state_dict(),
                "cov_act": cov_act.state_dict(),
                "optimizer": opt.state_dict(),
                "best_acc": best_acc,
            }, f"checkpoints/pid_arch_epoch{epoch+1}_acc{acc*100:.1f}.pt")

    wandb.finish()
    print(f"\nTraining complete. Best accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
