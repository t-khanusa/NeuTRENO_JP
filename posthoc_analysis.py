"""
Post-hoc analysis of training_neutreno_reg.py checkpoints under one fixed protocol.

Every checkpoint is measured on the same images, transforms and views:

  probe     frozen linear probe (L-BFGS logistic regression on standardized
            features, weight decay picked on a held-out 10% of train, then refit on
            all of train) and weighted k-NN (k=20, T=0.07, cosine after centring on
            the train mean); train and val both use the test transform
  spectrum  covariance spectrum of the val set (participation / entropy rank)
  views     a fixed set of val images × views (training multi-crop, seeded per
            image): spectrum of the per-image view centres and the within-image /
            between-image variance, so augmentation variance does not inflate rank
  tokens    mean pairwise cosine of patch tokens after every block (token
            over-smoothing), on the residual stream and after the LayerNorm that
            reads it next
  grads     per block: ‖∇ weighted invariance‖, ‖∇ weighted regularizer‖ and their
            cosine, on the same fixed training batches (train mode, seeded, so
            drop-path, BN batch statistics and the regularizer's random slices are
            identical for both terms)

Representations: raw CLS (384-d, input of the head), emb (512-d), proj (16-d), and
the CLS / mean patch token after blocks 3, 6, 9, 12 (pre final norm).

    python posthoc_analysis.py checkpoints/neutreno0.6_visreg/last.pt checkpoints/timm_visreg/last.pt
    python posthoc_analysis.py <ckpt>... --skip grads --out logs/posthoc

Writes <out>/<name>.json (summaries) and <out>/<name>.samples.pt (per-image /
per-batch values for the paired bootstraps in mechanism_claims.py).
"""

import argparse
import copy
import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.utils.data import DataLoader

from anticollapse import AntiCollapse
from training_lejepa import MultiCropDataset
from training_neutreno import NeuTRENOViT
from training_neutreno_reg import build_net, spectrum, view_geometry

LAYERS = (3, 6, 9, 12)
SECTIONS = ("probe", "spectrum", "views", "tokens", "grads")


# ─────────────────────────────────────────────────────────────────────────────
# Forward with intermediate tokens
# ─────────────────────────────────────────────────────────────────────────────

def forward_all(net, x, layers=LAYERS, tokens=False):
    """Representations of x (B, C, H, W): {name: (B, D)}, and with tokens=True the
    per-image patch-token cosine after every block, {layer: (pre, post) (B,)}:
    pre = residual stream, post = after the LayerNorm that reads it next (the next
    block's norm1, or the final norm after the last block)."""
    bb = net.backbone
    hidden = {}
    if isinstance(bb, NeuTRENOViT):
        h = bb.patch_embed(x)
        h = torch.cat((bb.cls_token.expand(h.shape[0], -1, -1), h), dim=1)
        h = bb.pos_drop(h + bb.pos_embed)
        h, v0 = bb.blocks[0](h)
        hidden[1] = h
        for i, blk in enumerate(bb.blocks[1:], start=2):
            h = blk(h, v0=v0)
            hidden[i] = h
    else:
        h = bb.norm_pre(bb.patch_drop(bb._pos_embed(bb.patch_embed(x))))
        for i, blk in enumerate(bb.blocks, start=1):
            h = blk(h)
            hidden[i] = h
    cls = bb.norm(h)[:, 0]
    emb = bb.head(cls)
    reps = {"cls": cls, "emb": emb, "proj": net.proj(emb)}
    for l in layers:
        reps[f"L{l}_cls"] = hidden[l][:, 0]
        reps[f"L{l}_patch"] = hidden[l][:, 1:].mean(1)
    tok = {}
    if tokens:
        depth = len(bb.blocks)
        for l, hl in hidden.items():
            post = bb.blocks[l].norm1(hl) if l < depth else bb.norm(hl)
            tok[l] = (token_cos(hl[:, 1:]), token_cos(post[:, 1:]))
    return reps, tok


def token_cos(patches):
    """Mean pairwise cosine between distinct patch tokens, per image (B,)."""
    t = F.normalize(patches.float(), dim=-1)
    L = t.shape[1]
    return (t @ t.transpose(1, 2)).sum((1, 2)).sub(L).div(L * (L - 1))


@torch.no_grad()
def extract(net, loader, device, layers=LAYERS):
    """Test-transform features of a whole split: reps (N, D) on CPU, labels, and the
    per-image token cosine {layer: (pre (N,), post (N,))} after every block."""
    feats, ys, pre, post = {}, [], {}, {}
    for vs, y in loader:
        with autocast(device, dtype=torch.bfloat16):
            reps, tok = forward_all(net, vs[:, 0].to(device, non_blocking=True), layers, tokens=True)
        for k, v in reps.items():
            feats.setdefault(k, []).append(v.float().cpu())
        for l, (a, b) in tok.items():
            pre.setdefault(l, []).append(a.cpu())
            post.setdefault(l, []).append(b.cpu())
        ys.append(y)
    return ({k: torch.cat(v) for k, v in feats.items()}, torch.cat(ys),
            {l: (torch.cat(pre[l]), torch.cat(post[l])) for l in pre})


# ─────────────────────────────────────────────────────────────────────────────
# Probes
# ─────────────────────────────────────────────────────────────────────────────

def _fit_logreg(x, y, n_cls, wd, iters):
    lin = nn.Linear(x.shape[1], n_cls).to(x.device)
    opt = torch.optim.LBFGS(lin.parameters(), lr=1, max_iter=iters,
                            history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lin(x), y) + wd * lin.weight.square().sum()
        loss.backward()
        return loss

    opt.step(closure)
    return lin


def linear_probe(xtr, ytr, xva, yva, device, wds=(0.0, 1e-5, 1e-4, 1e-3, 1e-2), iters=300):
    xtr, ytr, xva, yva = (t.to(device) for t in (xtr, ytr, xva, yva))
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    xtr, xva = (xtr - mu) / sd, (xva - mu) / sd
    n_cls = int(ytr.max()) + 1
    perm = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(0)).to(device)
    hold, fit = perm[: len(xtr) // 10], perm[len(xtr) // 10:]

    def acc(lin, x, y):
        with torch.no_grad():
            return (lin(x).argmax(1) == y).float().mean().item()

    held = {wd: acc(_fit_logreg(xtr[fit], ytr[fit], n_cls, wd, iters), xtr[hold], ytr[hold])
            for wd in wds}
    wd = max(held, key=held.get)
    return acc(_fit_logreg(xtr, ytr, n_cls, wd, iters), xva, yva), wd


@torch.no_grad()
def knn(xtr, ytr, xva, yva, device, k=20, T=0.07):
    xtr, ytr, xva, yva = (t.to(device) for t in (xtr, ytr, xva, yva))
    mu = xtr.mean(0)
    a, b = F.normalize(xtr - mu, dim=1), F.normalize(xva - mu, dim=1)
    top_v, top_i = (b @ a.T).topk(k, dim=1)
    votes = torch.zeros(len(b), int(ytr.max()) + 1, device=device)
    votes.scatter_add_(1, ytr[top_i], (top_v / T).exp())
    return (votes.argmax(1) == yva).float().mean().item()


# ─────────────────────────────────────────────────────────────────────────────
# Fixed views and batches
# ─────────────────────────────────────────────────────────────────────────────

def seeded_crops(cfg, split, n_images, seed):
    """(n_images, V, C, H, W) training multi-crops of a fixed image subset, seeded per image."""
    ds = MultiCropDataset(split, V_global=cfg.get("V_global", 2), V_local=cfg.get("V_local", 6),
                          img_size=cfg.get("img_size", 128),
                          dataset_name=cfg.get("dataset", "imagenette"),
                          global_scale=tuple(cfg.get("global_scale", (0.4, 1.0))))
    idx = torch.randperm(len(ds), generator=torch.Generator().manual_seed(seed))[:n_images]
    crops = []
    for i in idx.tolist():
        torch.manual_seed(seed * 1_000_003 + i)
        crops.append(ds[i][0])
    return torch.stack(crops)


def strip_rep(metrics, rep):
    """{'geom/participation_rank_<rep>_center': v} → {'participation_rank_center': v}"""
    return {m.split("/", 1)[1].replace(f"_{rep}", "", 1): v for m, v in metrics.items()}


@torch.no_grad()
def view_stats(net, crops, device, layers=LAYERS, chunk=32):
    N, V = crops.shape[:2]
    reps = {}
    for s in range(0, N, chunk):
        x = crops[s:s + chunk].to(device).flatten(0, 1)
        with autocast(device, dtype=torch.bfloat16):
            r, _ = forward_all(net, x, layers)
        for k, v in r.items():
            reps.setdefault(k, []).append(v.float().reshape(-1, V, v.shape[-1]).cpu())
    out, samples = {}, {}
    for k, v in reps.items():
        x = torch.cat(v).transpose(0, 1)                          # (V, N, D)
        out[k] = strip_rep(view_geometry(x, k), k)
        samples[k] = {"center": x.mean(0).half(), "within": x.var(0).mean(-1)}
    return out, samples


# ─────────────────────────────────────────────────────────────────────────────
# Per-term gradients
# ─────────────────────────────────────────────────────────────────────────────

def split_terms(cfg, loss, terms):
    """(weighted invariance part, weighted regularizer part) of repr_loss."""
    reg = str(cfg.get("reg"))
    inv = terms["inv"]
    w = {"sigreg": 1 - cfg.get("lamb", 0.02), "visreg": 1 - cfg.get("visreg_lamb", 0.6),
         "lambdajepa": cfg.get("lj_w_inv", 25.0), "sacreg": cfg.get("lj_w_inv", 25.0),
         "vicreg": cfg.get("vic_w_inv", 25.0)}
    if reg in w:
        inv_part = w[reg] * inv
    elif reg == "barlow":       # diagonal = Barlow's invariance
        inv_part = cfg.get("bt_scale", 1.0) * terms["bt_on"]
    elif reg == "uniformity":
        inv_part = terms["align"]
    else:
        inv_part = loss
    return inv_part, loss - inv_part


def param_group(name):
    if name.startswith("backbone.blocks."):
        return f"block{int(name.split('.')[2]) + 1:02d}"
    if name.startswith("proj."):
        return "proj"
    if name.startswith(("backbone.norm.", "backbone.head.", "backbone.fc_norm.")):
        return "head"
    return "embed"


def grad_split(net, cfg, batches, device):
    """Per param group: ‖g_inv‖, ‖g_reg‖, cos(g_inv, g_reg) — batch mean and per batch."""
    state = copy.deepcopy(net.state_dict())          # train-mode BN updates running stats
    net.train()
    anti = AntiCollapse(cfg, n_global=cfg.get("V_global", 2)).to(device)
    params = [(n, p) for n, p in net.named_parameters() if p.requires_grad]
    groups = sorted({param_group(n) for n, _ in params})
    per = {g: {"norm_inv": [], "norm_reg": [], "cos": []} for g in groups}
    for b, vs in enumerate(batches):
        vs = vs.to(device)
        grads = {}
        for part in ("inv", "reg"):
            torch.manual_seed(10_000 + b)
            with autocast(device, dtype=torch.bfloat16):
                emb, proj = net(vs)
            emb_v = emb.reshape(-1, vs.shape[1], emb.shape[-1]).transpose(0, 1)
            loss, terms = anti(proj, emb_v)
            target = split_terms(cfg, loss, terms)[part == "reg"]
            net.zero_grad(set_to_none=True)
            target.backward()
            grads[part] = {g: torch.cat([p.grad.flatten().float() for n, p in params
                                         if param_group(n) == g and p.grad is not None])
                           for g in groups}
        for g in groups:
            gi, gr = grads["inv"][g], grads["reg"][g]
            per[g]["norm_inv"].append(gi.norm().item())
            per[g]["norm_reg"].append(gr.norm().item())
            per[g]["cos"].append(F.cosine_similarity(gi, gr, dim=0).item())
    net.zero_grad(set_to_none=True)
    net.load_state_dict(state)
    net.eval()
    return ({g: {k: sum(v) / len(v) for k, v in d.items()} for g, d in per.items()}, per)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def load_ckpt(path):
    """Checkpoint + run cfg. The SIGReg runs (training_lejepa.py,
    training_neutreno_lejepa.py) saved no cfg; theirs is rebuilt from the saved
    flags and the neutreno_lejepa.yaml / lejepa.yaml values (LeJEPA data, ViT-S/8)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if "cfg" not in ck:
        neutreno = ck.get("use_neutreno", False)
        ck["cfg"] = {"reg": "sigreg" if ck.get("use_sigreg", True) else "none", "lamb": 0.02,
                     "backbone": "neutreno" if neutreno else "timm",
                     "lambda_p": float(ck.get("lambda_p", 0.6)) if neutreno else 0.0,
                     "dataset": "imagenette", "V_global": 2, "V_local": 6, "img_size": 128,
                     "proj_dim": 16, "embed_dim": 512, "bs": 128, "epochs": 800, "seed": 0}
    return ck


def ckpt_name(path, ck):
    c = ck["cfg"]
    mode = c.get("res_mode", "full")
    if c.get("backbone", "neutreno") != "neutreno":
        bb = "timm"
    elif mode == "scale":
        bb = f"resscale{float(c.get('res_alpha', 1.0))}"
    elif mode == "negdiag":
        bb = "neutreno-negdiag"
    else:
        bb = f"neutreno{float(c.get('lambda_p', 0.6))}" + ("" if mode == "full" else f"-{mode}")
    stem = os.path.splitext(os.path.basename(path))[0]
    return (f"{c.get('run_name') or bb + '_' + str(c.get('reg'))}_s{c.get('seed', 0)}"
            f"_{stem}_ep{ck['epoch'] + 1}")


def analyze(path, data, args, device):
    ck = load_ckpt(path)
    cfg = ck["cfg"]
    net = build_net(cfg, device)
    net.load_state_dict(ck["net"])
    net.eval()
    out = {"path": path, "name": ckpt_name(path, ck), "epoch": ck["epoch"] + 1,
           "online_acc": ck.get("acc"), "cfg": cfg}
    # per-sample quantities for paired bootstraps (mechanism_claims.py)
    samples = {"name": out["name"], "cfg": cfg, "path": path}

    if {"probe", "spectrum", "tokens"} & args.sections:
        ftr, ytr, _ = extract(net, data["train"], device)
        fva, yva, tok = extract(net, data["val"], device)
        samples["val"] = {k: v.half() for k, v in fva.items()}
        samples["tokens"] = tok
        if "probe" in args.sections:
            out["probe"] = {}
            for k in fva:
                # L-BFGS is sync-bound: ~40× faster on CPU than on a shared GPU
                lin, wd = linear_probe(ftr[k], ytr, fva[k], yva, "cpu")
                out["probe"][k] = {"linear": lin, "linear_wd": wd,
                                   "knn": knn(ftr[k], ytr, fva[k], yva, device)}
        if "spectrum" in args.sections:
            out["spectrum"] = {k: strip_rep(spectrum(fva[k], k), k) for k in fva}
        if "tokens" in args.sections:
            out["tokens"] = {f"L{l}": {"pre": a.mean().item(), "post": b.mean().item()}
                             for l, (a, b) in tok.items()}
    if "views" in args.sections:
        out["views"], samples["views"] = view_stats(net, data["crops"], device)
    if "grads" in args.sections:
        out["grads"], samples["grads"] = grad_split(net, cfg, data["batches"], device)
    return out, samples


def print_tables(results):
    names = [r["name"] for r in results]
    w = max(14, *(len(n) for n in names))

    def table(title, rows, cols, get, fmt="{:.3f}"):
        lw = max(len(f"{row} {col}".strip()) for row in rows for col in cols)
        print(f"\n## {title}")
        print(f"{'':{lw}s}" + "".join(f"  {n:>{w}s}" for n in names))
        for row in rows:
            for col in cols:
                vals = []
                for r in results:
                    try:
                        vals.append(fmt.format(get(r, row, col)))
                    except (KeyError, TypeError):
                        vals.append("-")
                print(f"{(row + ' ' + col).strip():{lw}s}" + "".join(f"  {v:>{w}s}" for v in vals))

    reps = list(results[0].get("probe") or results[0].get("spectrum") or results[0].get("views") or [])
    if "probe" in results[0]:
        table("frozen probe (val acc)", reps, ["linear", "knn"], lambda r, k, c: r["probe"][k][c])
    if "spectrum" in results[0]:
        table("val spectrum", reps, ["participation_rank", "entropy_rank"],
              lambda r, k, c: r["spectrum"][k][c], "{:.2f}")
    if "views" in results[0]:
        table("fixed views", reps, ["participation_rank", "participation_rank_center",
                                    "within_over_between"],
              lambda r, k, c: r["views"][k][c], "{:.2f}")
    if "tokens" in results[0]:
        table("patch token cosine", list(results[0]["tokens"]), ["pre", "post"],
              lambda r, k, c: r["tokens"][k][c])
    if "grads" in results[0]:
        table("grad inv vs reg", list(results[0]["grads"]), ["norm_inv", "norm_reg", "cos"],
              lambda r, k, c: r["grads"][k][c], "{:.3g}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--skip", nargs="*", default=[], choices=SECTIONS)
    ap.add_argument("--view-images", type=int, default=512)
    ap.add_argument("--grad-batches", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--out", default="logs/posthoc")
    args = ap.parse_args()
    args.sections = set(SECTIONS) - set(args.skip)
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False     # exact fp32 for probes/spectra

    # one data protocol for every checkpoint (taken from the first one's cfg)
    cfg = load_ckpt(args.ckpts[0])["cfg"]
    data = {}
    if {"probe", "spectrum", "tokens"} & args.sections:
        for split, key in (("train", "train"), ("validation", "val")):
            ds = MultiCropDataset(split, V_global=0, V_local=0, img_size=cfg.get("img_size", 128),
                                  dataset_name=cfg.get("dataset", "imagenette"))
            ds.V = 1
            data[key] = DataLoader(ds, batch_size=256, num_workers=args.num_workers, pin_memory=True)
    if "views" in args.sections:
        data["crops"] = seeded_crops(cfg, "validation", args.view_images, args.seed)
    if "grads" in args.sections:
        crops = seeded_crops(cfg, "train", args.grad_batches * cfg.get("bs", 128), args.seed + 1)
        data["batches"] = crops.split(cfg.get("bs", 128))

    os.makedirs(args.out, exist_ok=True)
    results = []
    for path in args.ckpts:
        t0 = time.time()
        r, samples = analyze(path, data, args, device)
        with open(os.path.join(args.out, f"{r['name']}.json"), "w") as f:
            json.dump(r, f, indent=1)
        torch.save(samples, os.path.join(args.out, f"{r['name']}.samples.pt"))
        print(f"[posthoc] {r['name']} done in {time.time() - t0:.0f}s → {args.out}/{r['name']}.json", flush=True)
        results.append(r)
    print_tables(results)


if __name__ == "__main__":
    main()
