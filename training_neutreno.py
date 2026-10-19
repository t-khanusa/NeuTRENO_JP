"""NeuTRENO-JEPA: does the anti-oversmoothing value residual prevent JEPA collapse?

NeuTRENO (Nguyen et al., NeurIPS 2024) derives self-attention as a gradient
descent step on a nonlocal total-variation functional J(u) whose minimizers are
CONSTANT token fields — this is why deep transformers over-smooth. Adding a
regularizer G(v) = λ/2 Σ_i ||v_i − v_i^0||² yields the corrected update

    u(ℓ+1) = Attn(u(ℓ)) + λ_p (V0 − V(ℓ))          (P-control on values)

i.e. every layer receives a proportional pull of its value vectors toward the
FIRST layer's values V0. It is the P-only special case of PIDformer.

What this tests for JEPA:
  * NeuTRENO fights TOKEN collapse (tokens within one image becoming equal
    along depth). JEPA collapse is SAMPLE collapse (different images mapping
    to one point). These are different fixed points: the invariance gradient
    can still shrink the batch covariance through (a) the patch_embed + block-0
    value projection that define V0 itself, and (b) the head/projector behind
    the anchored trunk — the usual unconstrained min-cut.
  * Hypothesis: λ_p delays and slows collapse (V0 keeps injecting
    input-dependent signal at every depth) but cannot remove the collapsed
    fixed point. The λ_p sweep measures how much architectural token-level
    control alone buys at the sample level.

Loss = multi-view invariance + online probe (probe on detached emb). NO
covariance loss, NO whitening, NO teacher/EMA/stop-grad, NO homeostat.

Usage:
    python training_neutreno.py --config-path configs --config-name neutreno
    python training_neutreno.py --config-path configs --config-name neutreno lambda_p=0.6
    bash grid_neutreno.sh          # sweep lambda_p incl. 0.0 baseline
"""

import os
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP

import timm
from timm.layers import PatchEmbed, Mlp, DropPath, trunc_normal_
import wandb
import hydra
import tqdm
from omegaconf import DictConfig, OmegaConf

from training_pid_sample import MultiCropDataset


# ─────────────────────────────────────────────────────────────────────────────
# NeuTRENO backbone — author mechanism, P-only value control:
#     layer 0:   keep V0
#     layer ℓ>0: x = Attn(x) + λ_p (V0 − V(ℓ))
# Parameter names match standard ViT → pretrained loading stays possible.
#
# res_mode isolates the two halves of the correction (layers ℓ>0, β = λ_p):
#     full    Attn(x) + β (V0 − V(ℓ))       NeuTRENO
#     anchor  Attn(x) + β V0                 anchor-only
#     sub     Attn(x) − β V(ℓ)               subtraction-only
#     scale   x + α · AttnBranch(x), no value term — plain scalar scaling of the
#             attention residual branch (α = res_alpha), the reparameterisation
#             NeuTRENO is often mistaken for
# ─────────────────────────────────────────────────────────────────────────────
RES_MODES = ("full", "anchor", "sub", "scale")
# ─────────────────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────

class NeuTRENOAttention(nn.Module):
    fused_attn = False   # opt-in F.scaled_dot_product_attention (set by training_neutreno_reg.py)

    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 layerth=0, lambda_p=0.6, res_mode="full"):
        super().__init__()
        assert dim % num_heads == 0
        assert res_mode in RES_MODES, res_mode
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.lambda_p = lambda_p     # NeuTRENO λ (paper default 0.6), fixed, not learned
        self.layerth = layerth
        self.res_mode = res_mode

    def forward(self, x, v0=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        if self.fused_attn:
            # same math as the explicit path (attn_drop must be 0), memory-efficient kernel
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            out = attn @ v

        if self.layerth == 0 or self.res_mode == "scale":
            res = 0.
        elif self.res_mode == "full":
            res = self.lambda_p * (v0 - v)
        elif self.res_mode == "anchor":
            res = self.lambda_p * v0
        else:
            res = -self.lambda_p * v
        x = out + res
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)

        if self.layerth == 0:
            return x, v
        return x


class NeuTRENOBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 layerth=0, lambda_p=0.6, res_mode="full", res_alpha=1.0):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = NeuTRENOAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                      attn_drop=attn_drop, proj_drop=drop,
                                      layerth=layerth, lambda_p=lambda_p, res_mode=res_mode)
        # residual-branch scale (res_mode=scale, layers ℓ>0); 1.0 everywhere else
        self.res_alpha = res_alpha if (res_mode == "scale" and layerth > 0) else 1.0
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio),
                       act_layer=act_layer, drop=drop)
        self.layerth = layerth

    def forward(self, x, v0=None):
        if self.layerth == 0:
            x_, v0 = self.attn(self.norm1(x))
        else:
            x_ = self.attn(self.norm1(x), v0=v0)
        if self.res_alpha != 1.0:
            x_ = self.res_alpha * x_
        x = x + self.drop_path(x_)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if self.layerth == 0:
            return x, v0
        return x


class NeuTRENOViT(nn.Module):
    def __init__(self, img_size=128, patch_size=8, in_chans=3, num_classes=512,
                 embed_dim=384, depth=12, num_heads=6, mlp_ratio=4., qkv_bias=True,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, act_layer=None, lambda_p=0.6, res_mode="full", res_alpha=1.0):
        super().__init__()
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                      in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            NeuTRENOBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate,
                          drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer,
                          layerth=i, lambda_p=lambda_p, res_mode=res_mode, res_alpha=res_alpha)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        self.grad_checkpointing = False

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x):
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.pos_drop(x + self.pos_embed)

        if self.grad_checkpointing and torch.is_grad_enabled():
            ckpt = partial(torch.utils.checkpoint.checkpoint, use_reentrant=False)
            x, v0 = ckpt(self.blocks[0], x)
            for i in range(1, len(self.blocks)):
                x = ckpt(self.blocks[i], x, v0)
        else:
            x, v0 = self.blocks[0](x)
            for i in range(1, len(self.blocks)):
                x = self.blocks[i](x, v0=v0)

        x = self.norm(x)
        return self.head(x[:, 0])


class NeuTRENOEncoder(nn.Module):
    """NeuTRENO backbone + the same MLP projector as every other pipeline."""

    def __init__(self, img_size=128, patch_size=8, embed_dim=384, depth=12,
                 num_heads=6, mlp_ratio=4.0, out_dim=512, proj_dim=16,
                 drop_path_rate=0.1, lambda_p=0.6, res_mode="full", res_alpha=1.0):
        super().__init__()
        self.backbone = NeuTRENOViT(
            img_size=img_size, patch_size=patch_size,
            embed_dim=embed_dim, depth=depth, num_heads=num_heads,
            mlp_ratio=mlp_ratio, num_classes=out_dim,
            drop_path_rate=drop_path_rate, qkv_bias=True, lambda_p=lambda_p,
            res_mode=res_mode, res_alpha=res_alpha,
        )
        self.proj = MLP(out_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Telemetry
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# Training (identical pipeline to the other invariance-only experiments)
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="neutreno")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    lambda_p = cfg.get("lambda_p", 0.6)
    run_name = f"neutreno_lp{lambda_p}" if lambda_p > 0 else "neutreno_lp0_baseline"
    wandb.init(project="PID-NeuTRENO", name=run_name,
               config=OmegaConf.to_container(cfg, resolve=True))
    torch.manual_seed(cfg.get("seed", 0))
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Dataset ────────────────────────────────────────────────────────────
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
    print(f"Backbone: NeuTRENO ViT (lambda_p={lambda_p})")
    net = NeuTRENOEncoder(
        img_size=img_size, patch_size=cfg.get("patch_size", 8),
        embed_dim=cfg.get("backbone_dim", 384), depth=cfg.get("depth", 12),
        num_heads=cfg.get("num_heads", 6), mlp_ratio=cfg.get("mlp_ratio", 4.0),
        out_dim=embed_dim, proj_dim=proj_dim,
        drop_path_rate=cfg.get("drop_path_rate", 0.1),
        lambda_p=lambda_p,
    ).to(device)

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

            log = {
                "train/inv_loss": inv_loss.item(),
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
                f"checkpoints/neutreno_lp{lambda_p}_epoch{epoch+1}_acc{acc*100:.1f}.pt")

    wandb.finish()
    print(f"\nTraining complete. Best accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
