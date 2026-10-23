"""
Frozen-backbone transfer in the protocol of VISReg Table 4 (Wu et al., 2606.02572,
"same settings as DINOv2" with 10-epoch probes), for NeuTRENO checkpoints.

Linear probe (7 in-domain + 3 OOD datasets)
    feature   concat of the final-norm CLS tokens of the last 4 blocks (4 × 384)
    head      BatchNorm + Linear, SGD momentum 0.9, no weight decay, batch 32,
              cosine over 10 epochs, 13 base lrs (the DINOv2 grid) × 32/256
    input     resized to R × R (no augmentation), ImageNet normalization
    metric    accuracy; macro AU-ROC over the 14 labels for NIH ChestX-ray14
    reported  best_test   best test metric over the 13 lrs (the paper's protocol)
              holdout     lr picked on a 10% split of train, head retrained on all
                          of train, test metric of that lr (no test selection)
Linear segmentation (ADE20K, 150 classes)
    head      BatchNorm2d + 1×1 conv on the final-norm patch tokens of block 12
    optim     AdamW 2e-3, wd 1e-4, poly decay 0.9 after 1500 linear warm-up iters,
              batch 16, 40 epochs; random rescale 0.5–2 / crop / flip
    metric    single-scale mIoU, sliding window, at full label resolution

Resolution. These ViT-S/8 models are pre-trained at 128 px (16×16 tokens); the
paper's ViT-B/16 is pre-trained and probed at 224 and segmented at 512 (32×32
tokens). Here classification runs at the native R=128 and segmentation at 256,
which gives the paper's 32×32 token grid (bicubic-interpolated position embeddings).
Galaxy10 and ChestX-ray14 splits are not given in the paper: Galaxy10 uses a
stratified 80/20 split (seed 0), ChestX-ray14 the official train_val / test lists.

    python downstream_eval.py checkpoints/neutreno0.0_visreg_imagenette_s0_ep300/last.pt \\
        checkpoints/neutreno0.6_visreg_imagenette_s0_ep300/last.pt
    python downstream_eval.py A.pt B.pt --tasks seg
"""

import argparse
import json
import math
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as tvd
from PIL import Image
from scipy.stats import rankdata
from torch.amp import autocast
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset
from torchvision.transforms import v2

from galaxy10 import Galaxy10Rows
from posthoc_analysis import ckpt_name, load_ckpt
from training_neutreno import NeuTRENOViT
from training_neutreno_reg import build_net

ROOT = "/project/khanhnt/datasets/downstream"
IN_DOMAIN = ["aircraft", "cars", "cifar10", "cifar100", "flowers", "food", "pets"]
OOD = ["dtd", "galaxy10", "chestxray"]
LRS = [1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1]
CXR_LABELS = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule",
              "Pneumonia", "Pneumothorax", "Consolidation", "Edema", "Emphysema", "Fibrosis",
              "Pleural_Thickening", "Hernia"]
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
PROBE_LAYERS = (9, 10, 11, 12)


# ─────────────────────────────────────────────────────────────────────────────
# Backbone at any resolution
# ─────────────────────────────────────────────────────────────────────────────

def load_backbone(path, device):
    ck = load_ckpt(path)
    if ck["cfg"].get("backbone", "neutreno") != "neutreno":
        raise NotImplementedError("downstream_eval.py handles NeuTRENO-ViT checkpoints")
    net = build_net(ck["cfg"], device)
    net.load_state_dict(ck["net"])
    bb = net.backbone.eval().requires_grad_(False)
    bb.patch_embed.strict_img_size = False
    return ckpt_name(path, ck), bb


def pos_embed_for(bb, gh, gw):
    pos = bb.pos_embed
    cls, grid = pos[:, :1], pos[:, 1:]
    g0 = int(math.isqrt(grid.shape[1]))
    if (gh, gw) == (g0, g0):
        return pos
    grid = grid.reshape(1, g0, g0, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid.float(), size=(gh, gw), mode="bicubic", align_corners=False,
                         antialias=gh < g0).to(pos.dtype)
    return torch.cat([cls, grid.flatten(2).transpose(1, 2)], dim=1)


def block_outputs(bb: NeuTRENOViT, x, layers):
    """Residual stream after each block in `layers` (1-based), (B, 1+L, C)."""
    p = bb.patch_embed.patch_size[0]
    gh, gw = x.shape[-2] // p, x.shape[-1] // p
    h = bb.patch_embed(x)
    h = torch.cat((bb.cls_token.expand(h.shape[0], -1, -1), h), dim=1)
    h = bb.pos_drop(h + pos_embed_for(bb, gh, gw))
    h, v0 = bb.blocks[0](h)
    out = {1: h} if 1 in layers else {}
    for i, blk in enumerate(bb.blocks[1:], start=2):
        h = blk(h, v0=v0)
        if i in layers:
            out[i] = h
        if i >= max(layers):
            break
    return out, (gh, gw)


# ─────────────────────────────────────────────────────────────────────────────
# Classification datasets
# ─────────────────────────────────────────────────────────────────────────────

class HFImages(Dataset):
    def __init__(self, ds, tf):
        self.ds, self.tf = ds, tf

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        r = self.ds[i]
        return self.tf(r["image"].convert("RGB")), r["label"]


class Galaxy10(Dataset):
    def __init__(self, root, split, tf):
        self.rows, self.tf = Galaxy10Rows(split, root), tf
        self.idx, self.y = self.rows.idx, self.rows.y

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        return self.tf(r["image"]), r["label"]


class ChestXray14(Dataset):
    def __init__(self, root, split, tf):
        import csv
        d = os.path.join(root, "chestxray14")
        with open(os.path.join(d, "train_val_list.txt" if split == "train" else "test_list.txt")) as f:
            names = [l.strip() for l in f if l.strip()]
        lab = {}
        with open(os.path.join(d, "Data_Entry_2017.csv")) as f:
            for r in csv.DictReader(f):
                lab[r["Image Index"]] = r["Finding Labels"].split("|")
        self.files = [os.path.join(d, "images", n) for n in names]
        self.y = torch.tensor([[c in lab[n] for c in CXR_LABELS] for n in names], dtype=torch.float32)
        self.tf = tf

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        return self.tf(Image.open(self.files[i]).convert("RGB")), self.y[i]


def cls_dataset(name, split, tf, root):
    """→ (dataset, num_classes, multilabel). split ∈ {train, test}."""
    tr = split == "train"
    p = lambda d: os.path.join(root, d)
    if name == "cifar10":
        return tvd.CIFAR10(p("cifar10"), train=tr, transform=tf), 10, False
    if name == "cifar100":
        return tvd.CIFAR100(p("cifar100"), train=tr, transform=tf), 100, False
    if name == "food":
        return tvd.Food101(p("food101"), split=split, transform=tf), 101, False
    if name == "flowers":
        ds = (ConcatDataset([tvd.Flowers102(p("flowers102"), split=s, transform=tf) for s in ("train", "val")])
              if tr else tvd.Flowers102(p("flowers102"), split="test", transform=tf))
        return ds, 102, False
    if name == "pets":
        return tvd.OxfordIIITPet(p("pets"), split="trainval" if tr else "test",
                                 target_types="category", transform=tf), 37, False
    if name == "dtd":
        ds = (ConcatDataset([tvd.DTD(p("dtd"), split=s, partition=1, transform=tf) for s in ("train", "val")])
              if tr else tvd.DTD(p("dtd"), split="test", partition=1, transform=tf))
        return ds, 47, False
    if name == "aircraft":
        return tvd.FGVCAircraft(p("aircraft"), split="trainval" if tr else "test",
                                annotation_level="variant", transform=tf), 100, False
    if name == "cars":
        from datasets import load_from_disk
        return HFImages(load_from_disk(p("stanford_cars"))[split], tf), 196, False
    if name == "galaxy10":
        return Galaxy10(root, split, tf), 10, False
    if name == "chestxray":
        return ChestXray14(root, split, tf), 14, True
    raise ValueError(name)


# ─────────────────────────────────────────────────────────────────────────────
# Features
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract(models, name, split, args, device):
    """Cached (x float16 (N, 4·384), y) per model for one dataset split."""
    paths = {m: os.path.join(args.out, "feats", m, f"{name}_{split}.pt") for m in models}
    todo = [m for m in models if not os.path.exists(paths[m])]
    if todo:
        tf = v2.Compose([v2.Lambda(lambda im: im.convert("RGB") if isinstance(im, Image.Image) else im),
                         v2.PILToTensor(), v2.Resize((args.res, args.res), antialias=True),
                         v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])
        ds, _, _ = cls_dataset(name, split, tf, args.root)
        if args.limit:
            ds = Subset(ds, np.random.RandomState(0).permutation(len(ds))[:args.limit].tolist())
        dl = DataLoader(ds, batch_size=256, num_workers=args.workers, pin_memory=True)
        feats, ys = {m: [] for m in todo}, []
        t0 = time.time()
        for x, y in dl:
            x = x.to(device, non_blocking=True)
            for m in todo:
                bb = models[m]
                with autocast("cuda", dtype=torch.bfloat16):
                    out, _ = block_outputs(bb, x, PROBE_LAYERS)
                    f = torch.cat([bb.norm(out[l])[:, 0] for l in PROBE_LAYERS], dim=1)
                feats[m].append(f.half().cpu())
            ys.append(y)
        y = torch.cat(ys)
        for m in todo:
            os.makedirs(os.path.dirname(paths[m]), exist_ok=True)
            torch.save({"x": torch.cat(feats[m]), "y": y}, paths[m])
        print(f"  features {name}/{split}: {len(y)} images, {time.time() - t0:.0f}s", flush=True)
    return {m: torch.load(paths[m]) for m in models}


# ─────────────────────────────────────────────────────────────────────────────
# 13 linear heads trained in parallel
# ─────────────────────────────────────────────────────────────────────────────

class StackedProbe(nn.Module):
    """H independent BatchNorm + Linear heads on the same features. Batch statistics
    are identical across heads, so only the BN affine is per head."""

    def __init__(self, H, D, C):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(H, 1, D))
        self.beta = nn.Parameter(torch.zeros(H, 1, D))
        self.W = nn.Parameter(torch.randn(H, D, C) * 0.01)
        self.b = nn.Parameter(torch.zeros(H, 1, C))
        self.register_buffer("rm", torch.zeros(D))
        self.register_buffer("rv", torch.ones(D))

    def forward(self, x):
        if self.training:
            m, v = x.mean(0), x.var(0, unbiased=False)
            with torch.no_grad():
                n = x.shape[0]
                self.rm.lerp_(m, 0.1)
                self.rv.lerp_(v * n / max(n - 1, 1), 0.1)
        else:
            m, v = self.rm, self.rv
        xn = (x - m) / (v + 1e-5).sqrt()
        return torch.baddbmm(self.b, xn * self.gamma + self.beta, self.W)   # (H, B, C)


def macro_auroc(scores, y):
    """scores, y: (N, C) numpy → mean over classes with both labels present."""
    aucs = []
    for c in range(y.shape[1]):
        pos = y[:, c] > 0.5
        P, N = pos.sum(), (~pos).sum()
        if P == 0 or N == 0:
            continue
        r = rankdata(scores[:, c])
        aucs.append((r[pos].sum() - P * (P + 1) / 2) / (P * N))
    return float(np.mean(aucs))


def train_probe(xtr, ytr, xte, yte, C, multilabel, args, device, seed=0, return_logits=False):
    """→ test metric (%) for each of the 13 lrs (and the (H, N, C) test logits)."""
    torch.manual_seed(seed)
    H, D = len(args.lrs), xtr.shape[1]
    xtr, xte = xtr.to(device).float(), xte.to(device).float()
    ytr = ytr.to(device)
    probe = StackedProbe(H, D, C).to(device)
    lr = torch.tensor(args.lrs, device=device) * args.probe_bs / 256
    opt = torch.optim.SGD(probe.parameters(), lr=1.0, momentum=0.9, weight_decay=0.0)
    n, bs = len(xtr), args.probe_bs
    steps = args.probe_epochs * (n // bs)
    g = torch.Generator(device="cpu").manual_seed(seed)
    t = 0
    probe.train()
    for _ in range(args.probe_epochs):
        perm = torch.randperm(n, generator=g).to(device)
        for s in range(n // bs):
            idx = perm[s * bs:(s + 1) * bs]
            logits = probe(xtr[idx])
            if multilabel:
                loss = F.binary_cross_entropy_with_logits(
                    logits, ytr[idx].float().expand(H, -1, -1), reduction="none").mean((1, 2)).sum()
            else:
                loss = F.cross_entropy(logits.flatten(0, 1), ytr[idx].repeat(H)) * H
            opt.zero_grad(set_to_none=True)
            loss.backward()
            # per-head lr: SGD(lr=1) on lr_h·g is SGD(lr_h) on g (no weight decay); the
            # cosine factor enters the momentum buffer one step early, a negligible lag
            cos = 0.5 * (1 + math.cos(math.pi * t / steps))
            for p in probe.parameters():
                p.grad.mul_(lr.view(H, *([1] * (p.dim() - 1))) * cos)
            opt.step()
            t += 1
    probe.eval()
    with torch.no_grad():
        logits = torch.cat([probe(xte[i:i + 4096]) for i in range(0, len(xte), 4096)], dim=1)
    if multilabel:
        s, y = torch.sigmoid(logits).float().cpu().numpy(), yte.numpy()
        metric = [100 * macro_auroc(s[h], y) for h in range(H)]
    else:
        metric = (logits.argmax(-1).cpu() == yte[None]).float().mean(1).mul(100).tolist()
    return (metric, logits.cpu()) if return_logits else metric


def eval_classification(models, args, device):
    res = {m: {} for m in models}
    for name in args.datasets:
        print(f"[cls] {name}", flush=True)
        tr = extract(models, name, "train", args, device)
        te = extract(models, name, "test", args, device)
        _, C, ml = cls_dataset(name, "test", None, args.root) if name != "chestxray" else (None, 14, True)
        for m in models:
            xtr, ytr = tr[m]["x"], tr[m]["y"]
            xte, yte = te[m]["x"], te[m]["y"]
            full = train_probe(xtr, ytr, xte, yte, C, ml, args, device)
            # lr selection without the test set: 10% of train as holdout
            perm = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(0))
            k = len(xtr) // 10
            hold = train_probe(xtr[perm[k:]], ytr[perm[k:]], xtr[perm[:k]], ytr[perm[:k]],
                               C, ml, args, device)
            h_sel = int(np.argmax(hold))
            res[m][name] = {"best_test": max(full), "best_lr": args.lrs[int(np.argmax(full))],
                            "holdout": full[h_sel], "holdout_lr": args.lrs[h_sel], "per_lr": full}
            print(f"  {m}: best_test={max(full):.2f} (lr {args.lrs[int(np.argmax(full))]:g})  "
                  f"holdout={full[h_sel]:.2f} (lr {args.lrs[h_sel]:g})", flush=True)
        merged = dump(res, args, "cls")
    return merged


# ─────────────────────────────────────────────────────────────────────────────
# ADE20K linear segmentation
# ─────────────────────────────────────────────────────────────────────────────

class ADE20K(Dataset):
    def __init__(self, root, split, size, train):
        d = os.path.join(root, "ADEChallengeData2016")
        sub = "training" if split == "train" else "validation"
        self.imgs = sorted(os.path.join(d, "images", sub, f)
                           for f in os.listdir(os.path.join(d, "images", sub)) if f.endswith(".jpg"))
        self.lbls = [p.replace("/images/", "/annotations/").replace(".jpg", ".png") for p in self.imgs]
        self.size, self.train = size, train
        self.norm = v2.Normalize(MEAN, STD)

    def __len__(self):
        return len(self.imgs)

    def _resize(self, img, lbl, short):
        w, h = img.size
        s = min(short / min(w, h), 4 * short / max(w, h))
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        return img.resize((nw, nh), Image.BILINEAR), lbl.resize((nw, nh), Image.NEAREST)

    def __getitem__(self, i):
        img = Image.open(self.imgs[i]).convert("RGB")
        lbl = Image.open(self.lbls[i])
        if self.train:
            img, lbl = self._resize(img, lbl, self.size * random.uniform(0.5, 2.0))
        else:
            img, lbl0 = self._resize(img, lbl, self.size)
        x = torch.from_numpy(np.asarray(img).copy()).permute(2, 0, 1).float().div(255)
        x = self.norm(x)
        # reduce_zero_label: 0 = "other" → ignore, classes 1..150 → 0..149
        y = torch.from_numpy(np.asarray(lbl).astype(np.int64)) - 1
        y[y < 0] = 255
        if not self.train:
            return x, y            # y at the original resolution
        S = self.size
        H, W = y.shape
        ph, pw = max(S - H, 0), max(S - W, 0)
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
            y = F.pad(y, (0, pw, 0, ph), value=255)
            H, W = y.shape
        t, l = random.randint(0, H - S), random.randint(0, W - S)
        x, y = x[:, t:t + S, l:l + S], y[t:t + S, l:l + S]
        if random.random() < 0.5:
            x, y = x.flip(-1), y.flip(-1)
        return x, y


def seg_features(bb, x):
    out, (gh, gw) = block_outputs(bb, x, (12,))
    f = bb.norm(out[12])[:, 1:]
    return f.transpose(1, 2).reshape(f.shape[0], -1, gh, gw)


@torch.no_grad()
def slide_logits(bb, head, x, S, n_cls=150):
    """Sliding-window logits (1, n_cls, H, W) for one image (1, 3, H, W)."""
    _, _, H, W = x.shape
    stride = S * 2 // 3
    hp, wp = max(S - H, 0), max(S - W, 0)
    x = F.pad(x, (0, wp, 0, hp))
    Hp, Wp = x.shape[-2:]
    ys = list(range(0, Hp - S, stride)) + [Hp - S]
    xs = list(range(0, Wp - S, stride)) + [Wp - S]
    acc = torch.zeros(1, n_cls, Hp, Wp, device=x.device)
    cnt = torch.zeros(1, 1, Hp, Wp, device=x.device)
    for t in ys:
        for l in xs:
            crop = x[..., t:t + S, l:l + S]
            with autocast("cuda", dtype=torch.bfloat16):
                f = seg_features(bb, crop)
            lg = F.interpolate(head(f.float()), size=(S, S), mode="bilinear", align_corners=False)
            acc[..., t:t + S, l:l + S] += lg
            cnt[..., t:t + S, l:l + S] += 1
    return (acc / cnt)[..., :H, :W]


def eval_segmentation(models, args, device):
    S, n_cls = args.seg_res, 150
    train_ds = ADE20K(args.root, "train", S, train=True)
    val_ds = ADE20K(args.root, "val", S, train=False)
    if args.limit:
        train_ds = Subset(train_ds, list(range(min(args.limit, len(train_ds)))))
        val_ds = Subset(val_ds, list(range(min(args.limit // 4 or 1, len(val_ds)))))
    dl = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True, num_workers=args.workers,
                    pin_memory=True, persistent_workers=True,
                    worker_init_fn=lambda w: random.seed(torch.initial_seed() % 2**32))
    D = next(iter(models.values())).norm.normalized_shape[0]
    heads = {m: nn.Sequential(nn.BatchNorm2d(D), nn.Conv2d(D, n_cls, 1)).to(device) for m in models}
    opts = {m: torch.optim.AdamW(heads[m].parameters(), lr=2e-3, weight_decay=1e-4) for m in models}
    total, warm = args.seg_epochs * len(dl), min(1500, args.seg_epochs * len(dl) // 10)
    sched = lambda t: (t + 1) / warm if t < warm else (1 - (t - warm) / max(total - warm, 1)) ** 0.9
    scheds = {m: torch.optim.lr_scheduler.LambdaLR(opts[m], sched) for m in models}
    t0 = time.time()
    for ep in range(args.seg_epochs):
        for h in heads.values():
            h.train()
        run = {m: 0.0 for m in models}
        for x, y in dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            for m, bb in models.items():
                with torch.no_grad(), autocast("cuda", dtype=torch.bfloat16):
                    f = seg_features(bb, x)
                lg = F.interpolate(heads[m](f.float()), size=y.shape[-2:], mode="bilinear",
                                   align_corners=False)
                loss = F.cross_entropy(lg, y, ignore_index=255)
                opts[m].zero_grad(set_to_none=True)
                loss.backward()
                opts[m].step()
                scheds[m].step()
                run[m] += loss.item()
        print(f"[seg] epoch {ep + 1}/{args.seg_epochs} "
              + " ".join(f"{m}={run[m] / len(dl):.3f}" for m in models)
              + f"  {time.time() - t0:.0f}s", flush=True)

    res = {}
    for m, bb in models.items():
        heads[m].eval()
        conf = torch.zeros(n_cls, n_cls, dtype=torch.long, device=device)
        for i in range(len(val_ds)):
            x, y = val_ds[i]
            x, y = x[None].to(device), y.to(device)
            lg = slide_logits(bb, heads[m], x, S)
            lg = F.interpolate(lg, size=y.shape, mode="bilinear", align_corners=False)
            pred, keep = lg.argmax(1)[0], y != 255
            conf += torch.bincount(y[keep] * n_cls + pred[keep], minlength=n_cls ** 2).view(n_cls, n_cls)
        inter = conf.diag().float()
        union = conf.sum(0).float() + conf.sum(1).float() - inter
        present = conf.sum(1) > 0
        miou = (inter[present] / union[present]).mean().item() * 100
        acc = (inter.sum() / conf.sum()).item() * 100
        res[m] = {"ade20k": {"mIoU": miou, "pixel_acc": acc}}
        print(f"  {m}: mIoU={miou:.2f}  aAcc={acc:.2f}", flush=True)
    return dump(res, args, "seg")


# ─────────────────────────────────────────────────────────────────────────────

def dump(res, args, kind):
    """Merge into <out>/<kind><tag>.json so runs on different datasets accumulate."""
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"{kind}{args.tag}.json")
    old = json.load(open(path)) if os.path.exists(path) else {}
    for m, r in res.items():
        old.setdefault(m, {}).update(r)
    with open(path, "w") as f:
        json.dump(old, f, indent=1)
    return old


def table(args):
    """Table-4 layout from everything accumulated in <out>/cls<tag>.json and seg<tag>.json."""
    load = lambda k: (json.load(open(p)) if os.path.exists(p := os.path.join(args.out, f"{k}{args.tag}.json"))
                      else {})
    cls_res, seg_res = load("cls"), load("seg")
    models = list(dict.fromkeys([*cls_res, *seg_res]))
    if not models:
        return
    cols = [d for d in IN_DOMAIN + OOD if any(d in cls_res.get(m, {}) for m in models)]
    seg = any("ade20k" in seg_res.get(m, {}) for m in models)
    w = max(len(m) for m in models) + 2
    for key in ("best_test", "holdout"):
        print(f"\nLinear probe ({key}; ChestX-ray = macro AU-ROC × 100)" + ("; ADE20K = mIoU" if seg else ""))
        print(" " * w + " ".join(f"{c[:9]:>9s}" for c in cols) + ("   ADE20K" if seg else ""))
        for m in models:
            r = cls_res.get(m, {})
            row = " ".join(f"{r[c][key]:9.2f}" if c in r else f"{'—':>9s}" for c in cols)
            s = seg_res.get(m, {}).get("ade20k")
            print(f"{m:{w}s}{row}" + ((f" {s['mIoU']:8.2f}" if s else f" {'—':>8s}") if seg else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--tasks", nargs="+", default=["cls", "seg"], choices=["cls", "seg"])
    ap.add_argument("--datasets", nargs="+", default=IN_DOMAIN + OOD)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--res", type=int, default=128)
    ap.add_argument("--seg-res", type=int, default=256)
    ap.add_argument("--probe-epochs", type=int, default=10)
    ap.add_argument("--probe-bs", type=int, default=32)
    ap.add_argument("--lrs", type=float, nargs="+", default=LRS, help="base lrs (× bs/256)")
    ap.add_argument("--seg-epochs", type=int, default=40)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="subsample each split (smoke tests)")
    ap.add_argument("--out", default="logs/downstream")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = "cuda"
    models = dict(load_backbone(p, device) for p in args.ckpts)
    print("models:", ", ".join(models), flush=True)
    if "cls" in args.tasks:
        eval_classification(models, args, device)
    if "seg" in args.tasks:
        eval_segmentation(models, args, device)
    table(args)


if __name__ == "__main__":
    main()
