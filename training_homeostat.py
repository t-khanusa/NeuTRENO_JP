"""Homeostatic-layer experiment: invariance-only JEPA, NO anti-collapse loss.

Hypothesis under test: distributed weight-space PID control (homeostat.py) on
the tail layers keeps every layer's output covariance at its setpoint, so the
invariance gradient has NO unregulated cut left to collapse through.

  * Loss = multi-view invariance + online probe (probe on detached emb).
    NO covariance loss, NO whitening actuator, NO teacher/EMA/stop-grad.
  * After every optimizer step, each controlled Linear layer applies its own
    local plasticity update  W <- W + kappa*(k_p*E + k_i*M)@W  where
    E = sigma2*I - Cov[activations].  See homeostat.py for the control law.
  * A/B: homeo_enabled=true vs false (pure inv-only baseline, collapses fast).

Decision curves (wandb):
  * geom/logdet_proj, geom/erank_proj      — the floor under test.
  * homeo/trace_ratio_min                  — worst-layer covariance vs setpoint
                                             (collapse ⇒ → 0; regulated ⇒ ≈ 1).
  * train/proj_var, train/inv_loss, test/acc.

Usage:
    python training_homeostat.py --config-path configs --config-name homeostat
    python training_homeostat.py --config-path configs --config-name homeostat homeo_enabled=false
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

from training_pid_sample import PIDformerEncoder, ViTEncoder, MultiCropDataset
from homeostat import HomeostaticController, collect_homeo_modules


def proj_geometry(proj_flat: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    """log det and effective rank of the projection batch covariance."""
    with torch.no_grad():
        z = proj_flat.detach().float()
        z = z - z.mean(dim=0, keepdim=True)
        c = (z.transpose(0, 1) @ z) / max(z.shape[0] - 1, 1)
        evals = torch.linalg.eigvalsh(0.5 * (c + c.transpose(0, 1)))
        evals = evals.clamp_min(eps)
        return {
            "geom/logdet_proj": evals.log().sum().item(),
            "geom/erank_proj": (evals.sum().square()
                                / evals.square().sum().clamp_min(1e-12)).item(),
            "geom/min_eval_proj": evals.min().item(),
        }


@hydra.main(version_base=None, config_path="configs", config_name="homeostat")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    enabled = cfg.get("homeo_enabled", True)
    run_name = (
        f"homeo_kp{cfg.get('homeo_k_p', 1.0)}_ki{cfg.get('homeo_k_i', 0.1)}"
        f"_kap{cfg.get('homeo_kappa', 1e-3)}_blk{cfg.get('homeo_last_blocks', 4)}"
        if enabled else "homeo_off_baseline"
    )
    wandb.init(project="PID-Homeostat", name=run_name,
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

    # ── Model ──────────────────────────────────────────────────────────────
    embed_dim = cfg.get("embed_dim", 512)
    proj_dim = cfg.proj_dim
    if cfg.get("token_pid", False):
        print("Backbone: PIDformer (token-level PID ON)")
        net = PIDformerEncoder(
            img_size=img_size, patch_size=cfg.get("patch_size", 8),
            embed_dim=384, depth=cfg.get("depth", 12),
            num_heads=cfg.get("num_heads", 6), mlp_ratio=cfg.get("mlp_ratio", 4.0),
            out_dim=embed_dim, proj_dim=proj_dim,
            drop_path_rate=cfg.get("drop_path_rate", 0.1),
            pretrained=cfg.get("pretrained", False),
        ).to(device)
    else:
        print("Backbone: Standard ViT from timm")
        net = ViTEncoder(
            model_name=cfg.get("model_name", "vit_small_patch8_224"),
            img_size=img_size, embed_dim=embed_dim, proj_dim=proj_dim,
            drop_path_rate=cfg.get("drop_path_rate", 0.1),
            pretrained=cfg.get("pretrained", False),
        ).to(device)

    probe = nn.Sequential(nn.LayerNorm(embed_dim),
                          nn.Linear(embed_dim, num_classes)).to(device)

    # ── Homeostatic controller (weight-space, after-step actuation) ────────
    homeo = None
    if enabled:
        modules = collect_homeo_modules(net, last_blocks=cfg.get("homeo_last_blocks", 4))
        print(f"Homeostatic layers ({len(modules)}):")
        for name, m in modules.items():
            print(f"  {name}  ({m.weight.shape[0]}x{m.weight.shape[1]})")
        homeo = HomeostaticController(
            modules,
            kappa=cfg.get("homeo_kappa", 1e-3),
            k_p=cfg.get("homeo_k_p", 1.0),
            k_i=cfg.get("homeo_k_i", 0.1),
            k_d=cfg.get("homeo_k_d", 0.0),
            eta_i=cfg.get("homeo_eta_i", 0.02),
            m_clip=cfg.get("homeo_m_clip", 5.0),
            ema=cfg.get("homeo_ema", 0.05),
            sigma2=cfg.get("homeo_sigma2", "auto"),
            warmup_steps=cfg.get("homeo_warmup", 100),
            max_rows=cfg.get("homeo_max_rows", 2048),
            g_max=cfg.get("homeo_g_max", 1.0),
        )

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

            proj = proj.float()
            Vv, Nn, P = proj.shape
            proj_flat = proj.reshape(Vv * Nn, P)

            # The ONLY representation loss: multi-view invariance.
            inv_loss = (proj.mean(0) - proj).square().mean()

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

            # Slow outer loop: local homeostatic plasticity on the weights.
            if homeo is not None:
                homeo.step()

            log = {
                "train/inv_loss": inv_loss.item(),
                "train/probe_loss": probe_loss.item(),
                "train/total_loss": loss.item(),
                "train/batch_var": batch_var.item(),
                "train/proj_var": proj_var.item(),
            }
            log.update(proj_geometry(proj_flat))
            if homeo is not None:
                log.update(homeo.metrics())
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
            ckpt = {
                "epoch": epoch, "net": net.state_dict(), "probe": probe.state_dict(),
                "optimizer": opt.state_dict(), "best_acc": best_acc,
            }
            if homeo is not None:
                ckpt["homeo"] = homeo.state_dict()
            torch.save(ckpt, f"checkpoints/homeostat_epoch{epoch+1}_acc{acc*100:.1f}.pt")

    wandb.finish()
    print(f"\nTraining complete. Best accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
