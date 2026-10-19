"""Input-Anchored PID ViT — collapse prevention by architecture, no loss term.

Idea (control theory + information theory):
  Treat the residual stream h_l of a ViT as the STATE of a dynamical system
  evolving along DEPTH l.  Take the input patch-embedding r = PatchEmbed(x) as a
  fixed REFERENCE (setpoint) and let a PID law pull every layer toward it:

      e_l = r - h_l                              (anchor error at depth l)
      I_l = I_{l-1} + e_l                        (integral along depth)
      D_l = e_l - e_{l-1}                         (derivative along depth)
      u_l = K_P e_l + K_I I_l + K_D D_l
      h_{l+1} = Block(h_l) + λ · u_l

  Why this fights collapse WITHOUT any anti-collapse loss:
    * Information theory / DPI: a deterministic layer cannot CREATE I(x;z).
      Collapse is I(x;z) → 0.  The anchor RE-INJECTS x at every layer — it is a
      persistent source of input information, so I(x;z_l) keeps a floor by
      construction (like a skip connection from the input to every block).
    * Control theory: r is an EXOGENOUS setpoint (detached).  Unlike a forward
      whitening actuator whose authority B ∝ q (within-view spread) vanishes at
      the operating point, the anchor force is ∝ (r - h) and never vanishes.

  Tension (honest):  invariance wants z(view1)=z(view2) but the anchor pulls
  toward r(crop_k), which DIFFER across crops → an invariance floor ∝ λ².
  Mitigation: anchor PATCH tokens only (keep per-token input info) and read the
  representation from the FREE CLS token (readout=cls), which can still become
  view-invariant while attending to the anchored patches.  readout=mean gives
  the stronger (directly anchored) but lower-invariance variant.

Loss = multi-view invariance + online probe ONLY.  No covariance loss, no
whitening, no teacher/EMA/stop-grad.  A/B against anchor_enabled=false.

Usage:
    python training_pid_anchor.py --config-path configs --config-name pid_anchor
    python training_pid_anchor.py --config-path configs --config-name pid_anchor anchor_enabled=false
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP

import timm
import wandb
import hydra
import tqdm
from omegaconf import DictConfig, OmegaConf

from training_pid_sample import MultiCropDataset


# ─────────────────────────────────────────────────────────────────────────────
# Input-Anchored PID ViT encoder (standard timm ViT + depth-PID toward input)
# ─────────────────────────────────────────────────────────────────────────────

class AnchoredViTEncoder(nn.Module):
    """Standard timm ViT whose residual stream is PID-controlled toward the input
    patch-embedding reference along depth. Isolates the input-anchor mechanism
    (no PIDformer token-PID)."""

    def __init__(self, model_name="vit_small_patch8_224", img_size=128,
                 embed_dim=512, proj_dim=16, drop_path_rate=0.1, pretrained=False,
                 anchor_enabled=True, k_p=0.1, k_i=0.02, k_d=0.02, authority=1.0,
                 anchor_tokens="patch", readout="cls", anchor_detach=True):
        super().__init__()
        self.vit = timm.create_model(
            model_name, pretrained=pretrained, num_classes=0, global_pool="",
            img_size=img_size, drop_path_rate=drop_path_rate,
        )
        vit_dim = self.vit.embed_dim
        self.head = nn.Linear(vit_dim, embed_dim)
        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

        self.anchor_enabled = anchor_enabled
        self.k_p = k_p
        self.k_i = k_i
        self.k_d = k_d
        self.authority = authority
        self.anchor_tokens = anchor_tokens      # "patch" (exclude cls) | "all"
        self.readout = readout                  # "cls" | "mean"
        self.anchor_detach = anchor_detach
        self.num_prefix = getattr(self.vit, "num_prefix_tokens", 1)

    def _embed_tokens(self, imgs):
        v = self.vit
        x = v.patch_embed(imgs)
        # _pos_embed adds the cls/prefix tokens and positional embedding.
        if hasattr(v, "_pos_embed"):
            x = v._pos_embed(x)
        else:
            cls = v.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls, x), dim=1) + v.pos_embed
        if hasattr(v, "patch_drop"):
            x = v.patch_drop(x)
        if hasattr(v, "norm_pre"):
            x = v.norm_pre(x)
        return x

    def _run_blocks(self, x):
        v = self.vit
        if not self.anchor_enabled:
            for blk in v.blocks:
                x = blk(x)
            return v.norm(x), {}

        r = x.detach() if self.anchor_detach else x     # exogenous setpoint = input
        # Mask the prefix (cls) slot out of the anchor when anchoring patches only.
        mask = torch.ones_like(r)
        if self.anchor_tokens == "patch":
            mask[:, : self.num_prefix, :] = 0.0

        # Only maintain integral / derivative memory when their gains are nonzero.
        # P-only (k_i=k_d=0) then avoids accumulating a depth-length autograd graph.
        maintain_i = self.k_i != 0.0
        maintain_d = self.k_d != 0.0
        integ = torch.zeros_like(x) if maintain_i else None
        e_prev = None
        anchor_err_sum = 0.0
        for blk in v.blocks:
            h = blk(x)
            e = (r - h) * mask
            u = self.k_p * e
            if maintain_i:
                integ = integ + e
                u = u + self.k_i * integ
            if maintain_d:
                d = e - e_prev if e_prev is not None else torch.zeros_like(e)
                u = u + self.k_d * d
                e_prev = e
            x = h + self.authority * u
            with torch.no_grad():
                anchor_err_sum += e.float().square().mean().item()

        diag = {"anchor/err_mean_depth": anchor_err_sum / max(len(v.blocks), 1)}
        with torch.no_grad():
            e_final = ((r - x) * mask).float()
            denom = (r * mask).float().square().mean().clamp_min(1e-8)
            diag["anchor/rel_err_final"] = (e_final.square().mean() / denom).item()
        return v.norm(x), diag

    def forward(self, imgs):
        """imgs: (N, V, C, H, W). Returns (emb, proj, diag)."""
        N, V = imgs.shape[:2]
        x = self._embed_tokens(imgs.flatten(0, 1))
        tokens, diag = self._run_blocks(x)

        if self.readout == "mean":
            pooled = tokens[:, self.num_prefix:].mean(dim=1)
        else:
            pooled = tokens[:, 0]

        emb = self.head(pooled)
        proj = self.proj(emb).reshape(N, V, -1).transpose(0, 1)
        return emb, proj, diag


def proj_geometry(proj_flat: torch.Tensor, eps: float = 1e-8) -> dict[str, float]:
    """log det and effective rank of the projection batch covariance."""
    with torch.no_grad():
        z = proj_flat.detach().float()
        z = z - z.mean(dim=0, keepdim=True)
        c = (z.transpose(0, 1) @ z) / max(z.shape[0] - 1, 1)
        evals = torch.linalg.eigvalsh(0.5 * (c + c.transpose(0, 1))).clamp_min(eps)
        return {
            "geom/logdet_proj": evals.log().sum().item(),
            "geom/erank_proj": (evals.sum().square()
                                / evals.square().sum().clamp_min(1e-12)).item(),
            "geom/min_eval_proj": evals.min().item(),
        }


@hydra.main(version_base=None, config_path="configs", config_name="pid_anchor")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    if not cfg.get("anchor_enabled", True):
        run_name = "baseline_noanchor"
    else:
        ctrl = "P" + ("I" if cfg.get("anchor_k_i", 0) else "") + ("D" if cfg.get("anchor_k_d", 0) else "")
        g_eff = cfg.get("anchor_authority", 1.0) * cfg.get("anchor_k_p", 0.1)
        run_name = (f"{ctrl}_g{g_eff:.3g}_{cfg.get('anchor_tokens', 'patch')}"
                    f"_{cfg.get('readout', 'cls')}")
    run_name = cfg.get("run_name", None) or run_name
    wandb.init(project="PID-Anchor", name=run_name,
               config=OmegaConf.to_container(cfg, resolve=True))
    torch.manual_seed(cfg.get("seed", 0))
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    embed_dim = cfg.get("embed_dim", 512)
    proj_dim = cfg.proj_dim
    net = AnchoredViTEncoder(
        model_name=cfg.get("model_name", "vit_small_patch8_224"),
        img_size=img_size, embed_dim=embed_dim, proj_dim=proj_dim,
        drop_path_rate=cfg.get("drop_path_rate", 0.1),
        pretrained=cfg.get("pretrained", False),
        anchor_enabled=cfg.get("anchor_enabled", True),
        k_p=cfg.get("anchor_k_p", 0.1),
        k_i=cfg.get("anchor_k_i", 0.02),
        k_d=cfg.get("anchor_k_d", 0.02),
        authority=cfg.get("anchor_authority", 1.0),
        anchor_tokens=cfg.get("anchor_tokens", "patch"),
        readout=cfg.get("readout", "cls"),
        anchor_detach=cfg.get("anchor_detach", True),
    ).to(device)
    print(f"Anchor enabled: {net.anchor_enabled} | tokens={net.anchor_tokens} "
          f"| readout={net.readout} | Kp={net.k_p} Ki={net.k_i} Kd={net.k_d} "
          f"λ={net.authority}")

    probe = nn.Sequential(nn.LayerNorm(embed_dim),
                          nn.Linear(embed_dim, num_classes)).to(device)

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
        for vs, y in tqdm.tqdm(train_loader, total=len(train_loader)):
            vs = vs.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            with autocast(device, dtype=torch.bfloat16):
                emb, proj, adiag = net(vs)

            proj = proj.float()
            Vv, Nn, P = proj.shape
            proj_flat = proj.reshape(Vv * Nn, P)

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

            log = {
                "train/inv_loss": inv_loss.item(),
                "train/probe_loss": probe_loss.item(),
                "train/total_loss": loss.item(),
                "train/batch_var": batch_var.item(),
                "train/proj_var": proj_var.item(),
            }
            log.update(proj_geometry(proj_flat))
            log.update(adiag)
            wandb.log(log)

        net.eval()
        probe.eval()
        correct = 0
        with torch.inference_mode():
            for vs, y in test_loader:
                vs = vs.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                with autocast(device, dtype=torch.bfloat16):
                    emb, _, _ = net(vs)
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
            }, f"checkpoints/pid_anchor_epoch{epoch+1}_acc{acc*100:.1f}.pt")

    wandb.finish()
    print(f"\nTraining complete. Best accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
