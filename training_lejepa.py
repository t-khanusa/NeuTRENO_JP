"""
LeJEPA Training with Local/Global Multi-Crop Views.

This implements the LeJEPA (Latent Energy Joint Embedding Predictive Architecture)
self-supervised pretraining with a standard ViT backbone and multi-crop strategy:
  - Global views: large crops capturing holistic structure
  - Local views: small crops capturing fine-grained details

The SIGReg loss prevents sample-level embedding collapse (all inputs → same output),
while the invariance loss enforces multi-view consistency.

Usage:
    python training_lejepa.py --config-path configs --config-name lejepa

    Or with inline overrides:
    python training_lejepa.py +lamb=0.02 +V_global=2 +V_local=4 +proj_dim=16 \
        +lr=2e-3 +bs=256 +epochs=800
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torchvision.ops import MLP

import timm
import wandb
import hydra
import tqdm
from omegaconf import DictConfig, OmegaConf
from datasets import load_dataset


# ─────────────────────────────────────────────────────────────────────────────
# SIGReg: Sliced Integral Gaussian Regularizer
# Prevents sample-level collapse by enforcing marginal Gaussianity of
# projected representations via empirical characteristic function matching.
# ─────────────────────────────────────────────────────────────────────────────

class SIGReg(nn.Module):
    def __init__(self, knots=17):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3 / (knots - 1)
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, proj):
        A = torch.randn(proj.size(-1), 256, device=proj.device)
        A = A.div_(A.norm(p=2, dim=0))
        x_t = (proj @ A).unsqueeze(-1) * self.t
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * proj.size(-2)
        return statistic.mean()


# ─────────────────────────────────────────────────────────────────────────────
# Encoder: Standard ViT (timm) + Projector
# ─────────────────────────────────────────────────────────────────────────────

class ViTEncoder(nn.Module):
    """
    Standard ViT encoder from timm with an MLP projector head.
    Uses dynamic_img_size to support multi-resolution inputs (local/global crops).
    """

    def __init__(self, model_name="vit_small_patch8_224", img_size=128,
                 embed_dim=512, proj_dim=16, drop_path_rate=0.1, pretrained=False):
        super().__init__()
        kwargs = dict(
            pretrained=pretrained,
            num_classes=embed_dim,
            drop_path_rate=drop_path_rate,
        )
        if "resnet" not in model_name.lower():
            kwargs["img_size"] = img_size
        self.backbone = timm.create_model(model_name, **kwargs)
        self.proj = MLP(embed_dim, [2048, 2048, proj_dim], norm_layer=nn.BatchNorm1d)

    def forward(self, x):
        N, V = x.shape[:2]
        emb = self.backbone(x.flatten(0, 1))
        return emb, self.proj(emb).reshape(N, V, -1).transpose(0, 1)


# ─────────────────────────────────────────────────────────────────────────────
# Multi-Crop Dataset with Local/Global Views
# ─────────────────────────────────────────────────────────────────────────────

class MultiCropDataset(torch.utils.data.Dataset):
    """
    Dataset producing multi-crop augmented views per image:
      - V_global large/global crops (scale global_scale, default 0.4-1.0)
      - V_local  small/local crops  (scale 0.05-0.4)

    All crops are resized to the same resolution (img_size) so they can be
    batched together, matching the MINIMAL.md approach.
    """

    def __init__(self, split, V_global=2, V_local=4, img_size=128,
                 dataset_name="imagenette", global_scale=(0.4, 1.0)):
        self.V_global = V_global
        self.V_local = V_local
        self.V = V_global + V_local

        if dataset_name == "imagenette":
            self.ds = load_dataset("frgfm/imagenette", "160px", split=split)
        elif dataset_name == "galaxy10":
            from galaxy10 import Galaxy10Rows
            self.ds = Galaxy10Rows(split)
        elif dataset_name == "imagenet":
            self.ds = load_dataset("ILSVRC/imagenet-1k", split=split)
        else:
            self.ds = load_dataset(dataset_name, split=split)

        # Global crop: large area, strong augmentation
        self.global_aug = v2.Compose([
            v2.RandomResizedCrop(img_size, scale=tuple(global_scale)),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))]),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        # Local crop: small area, lighter augmentation
        self.local_aug = v2.Compose([
            v2.RandomResizedCrop(img_size, scale=(0.05, 0.4)),
            v2.RandomApply([v2.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            v2.RandomGrayscale(p=0.2),
            v2.RandomApply([v2.GaussianBlur(kernel_size=7, sigma=(0.1, 2.0))], p=0.5),
            v2.RandomApply([v2.RandomSolarize(threshold=128)], p=0.2),
            v2.RandomHorizontalFlip(),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.test_transform = v2.Compose([
            v2.Resize(img_size),
            v2.CenterCrop(img_size),
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __getitem__(self, i):
        item = self.ds[i]
        img = item["image"].convert("RGB")
        label = item["label"]

        if self.V > 1:
            views = [self.global_aug(img) for _ in range(self.V_global)]
            views += [self.local_aug(img) for _ in range(self.V_local)]
            return torch.stack(views), label
        else:
            return torch.stack([self.test_transform(img)]), label

    def __len__(self):
        return len(self.ds)


# ─────────────────────────────────────────────────────────────────────────────
# Main Training Loop
# ─────────────────────────────────────────────────────────────────────────────

@hydra.main(version_base=None, config_path="configs", config_name="lejepa")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    wandb.init(project="LeJEPA", config=OmegaConf.to_container(cfg, resolve=True))
    torch.manual_seed(cfg.get("seed", 0))
    torch.backends.cudnn.benchmark = True

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Dataset ──────────────────────────────────────────────────────────
    V_global = cfg.get("V_global", 2)
    V_local = cfg.get("V_local", 4)
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
    test_ds.V = 1  # single view for test

    train_loader = DataLoader(
        train_ds, batch_size=cfg.bs, shuffle=True,
        drop_last=True, num_workers=cfg.get("num_workers", 8), pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, num_workers=cfg.get("num_workers", 8), pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────
    embed_dim = cfg.get("embed_dim", 512)
    net = ViTEncoder(
        model_name=cfg.get("model_name", "vit_small_patch8_224"),
        img_size=img_size,
        embed_dim=embed_dim,
        proj_dim=cfg.proj_dim,
        drop_path_rate=cfg.get("drop_path_rate", 0.1),
        pretrained=cfg.get("pretrained", False),
    ).to(device)

    probe = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes)).to(device)
    sigreg = SIGReg().to(device)

    # ── Optimizer & Scheduler ────────────────────────────────────────────
    g1 = {"params": net.parameters(), "lr": cfg.lr, "weight_decay": 5e-2}
    g2 = {"params": probe.parameters(), "lr": 1e-3, "weight_decay": 1e-7}
    opt = torch.optim.AdamW([g1, g2])

    warmup_steps = len(train_loader)
    total_steps = len(train_loader) * cfg.epochs
    s1 = LinearLR(opt, start_factor=0.01, total_iters=warmup_steps)
    s2 = CosineAnnealingLR(opt, T_max=total_steps - warmup_steps, eta_min=1e-3)
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

                # LeJEPA loss: invariance + SIGReg
                inv_loss = (proj.mean(0) - proj).square().mean()
                sigreg_loss = sigreg(proj)
                lejepa_loss = sigreg_loss * cfg.lamb + inv_loss * (1 - cfg.lamb)

                # Online linear probe
                y_rep = y.repeat_interleave(V)
                yhat = probe(emb.detach())
                probe_loss = F.cross_entropy(yhat, y_rep)

                loss = lejepa_loss + probe_loss

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            wandb.log({
                "train/probe": probe_loss.item(),
                "train/lejepa": lejepa_loss.item(),
                "train/sigreg": sigreg_loss.item(),
                "train/inv": inv_loss.item(),
                "train/batch_var": batch_var.item(),
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
            }, f"checkpoints/lejepa_epoch{epoch+1}_acc{acc*100:.1f}.pt")

    wandb.finish()
    print(f"\nTraining complete. Best accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
