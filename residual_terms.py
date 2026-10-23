"""
What does each part of the NeuTRENO update do to the residual stream it is added to?

For every block ℓ ≥ 2 the update of the residual stream h is split into

  ctx   W_O A V(ℓ)          token mixing (the plain attention branch)
  sub   −λ W_O V(ℓ)         self subtraction       ┐ NeuTRENO: ctx + sub + anc
  anc   +λ W_O V0           anchor to block-1 V    ┘
  mlp   MLP(norm2(h + attention branch))
  attn  the attention branch the arm actually adds (ctx + its own sub / anc, × α)
  block attn + mlp

with λ = 0.6. For arms that do not use sub / anc they are counterfactual: the same
weights, but the term is not in the forward pass (marked "cf"). negdiag: sub is its
learned −λ_h W_O V (per head). corner: ctx is the (1−λ) W_O A V it adds, next to a
+λ W_O V self copy (= −sub, which stays counterfactual).

On a fixed set of val images × training multi-crop views (posthoc_analysis.py
protocol), each term t is compared with the stream h it is added to, for the CLS
token and the mean patch token:

  ρ_w   = −Σ⟨t_w, h_w⟩ / Σ‖h_w‖²   view deviations from the image centre
  ρ_b   = −Σ⟨t_b, h_b⟩ / Σ‖h_b‖²   image centres from the dataset centre
  ρ_tok = −Σ⟨t_i, h_i − h̄⟩ / Σ‖h_i − h̄‖²   patch tokens from their image's mean token
          (ρ > 0: the term cancels that component of the stream; < 0: amplifies it)
  Δlog r = log r(h + t) − log r(h),  r = within-image / between-image variance
          (the ratio of mechanism_claims.py; < 0: the term alone makes views of an
          image closer relative to the spread of images)

and diag(A), the attention self-weight: sub turns it into diag(A) − λ.
95% bootstrap over images for Δlog r.

    python residual_terms.py imagenette timm=checkpoints/timm_visreg/last.pt \\
        full=checkpoints/neutreno0.6_visreg/last.pt sub=... --out logs/posthoc/terms
"""

import argparse
import json
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from posthoc_analysis import load_ckpt, seeded_crops
from training_neutreno import NeuTRENOViT
from training_neutreno_reg import build_net

LAM = 0.6
TERMS = ("ctx", "sub", "anc", "mlp", "attn", "block")
UNITS = ("cls", "patch")


def heads(at, u):
    B, N, C = u.shape
    qkv = at.qkv(u).reshape(B, N, 3, at.num_heads, C // at.num_heads).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    if hasattr(at, "q_norm"):
        q, k = at.q_norm(q), at.k_norm(k)
    return ((q * at.scale) @ k.transpose(-2, -1)).softmax(-1), v


def out_proj(at, t):
    """(B, H, N, d) head space → (B, N, C) through W_O, without the bias."""
    B, H, N, d = t.shape
    return t.transpose(1, 2).reshape(B, N, H * d) @ at.proj.weight.T


def arm_of(bb):
    if not isinstance(bb, NeuTRENOViT):
        return "timm", 0.0, 1.0
    at, blk = bb.blocks[1].attn, bb.blocks[1]
    return at.res_mode, at.lambda_p, blk.res_alpha


@torch.no_grad()
def terms(net, x):
    """Stream h before every block ℓ ≥ 2 and the terms added to it, pooled to CLS / mean patch;
    per-crop token-level ρ sums; attention self-weights. Checked against the module forward."""
    bb = net.backbone
    mode, lam, _ = arm_of(bb)
    if isinstance(bb, NeuTRENOViT):
        h = bb.patch_embed(x)
        h = torch.cat((bb.cls_token.expand(h.shape[0], -1, -1), h), dim=1) + bb.pos_embed
    else:
        h = bb.norm_pre(bb.patch_drop(bb._pos_embed(bb.patch_embed(x))))
        assert all(isinstance(b.ls1, nn.Identity) and isinstance(b.attn.norm, nn.Identity)
                   for b in bb.blocks)
    pooled, tok, diag, v0 = {}, {}, {}, None
    e = h
    ed = e[:, 1:] - e[:, 1:].mean(1, keepdim=True)
    pooled[(0, "h")] = {"cls": e[:, 0], "patch": e[:, 1:].mean(1)}
    for i, blk in enumerate(bb.blocks):
        at = blk.attn
        a, v = heads(at, blk.norm1(h))
        if i == 0:
            v0 = v
        t = {"ctx": out_proj(at, a @ v), "sub": out_proj(at, -LAM * v), "anc": out_proj(at, LAM * v0)}
        branch = t["ctx"]
        if i > 0 and lam > 0 and mode in ("full", "sub"):
            branch = branch + t["sub"] * (lam / LAM)
        if i > 0 and lam > 0 and mode in ("full", "anchor"):
            branch = branch + t["anc"] * (lam / LAM)
        if i > 0 and mode == "negdiag":          # sub = the learned −λ_h W_O V, not the λ counterfactual
            t["sub"] = out_proj(at, -at.res_lambda.view(1, -1, 1, 1) * v)
            branch = branch + t["sub"]
        if i > 0 and mode == "corner":           # ((1−λ)A + λI)V: ctx at weight 1−λ, plus +λ W_O V = −sub
            t["ctx"] = (1 - lam) * t["ctx"]
            branch = t["ctx"] - t["sub"] * (lam / LAM)
        alpha = getattr(blk, "res_alpha", 1.0)       # scale arm: α on blocks ℓ > 1 only
        a_out = alpha * (branch + at.proj.bias)
        t["ctx"] = alpha * t["ctx"]
        h1 = h + a_out
        t["mlp"] = blk.mlp(blk.norm2(h1))
        t["attn"], t["block"] = a_out, a_out + t["mlp"]    # what the arm actually adds
        if isinstance(bb, NeuTRENOViT):
            ref = blk(h)[0] if i == 0 else blk(h, v0=v0)
        else:
            ref = blk(h)
        new = h1 + t["mlp"]
        err = (new - ref).abs().max().item() / ref.abs().max().item()
        assert err < 1e-3, (i, err)
        if i > 0:
            l = i + 1
            hp = {"cls": h[:, 0], "patch": h[:, 1:].mean(1)}
            pooled[(l, "h")] = hp
            hd = h[:, 1:] - h[:, 1:].mean(1, keepdim=True)
            den = hd.pow(2).sum((1, 2))
            for k in TERMS:
                pooled[(l, k)] = {"cls": t[k][:, 0], "patch": t[k][:, 1:].mean(1)}
                tok[(l, k)] = (-(t[k][:, 1:] * hd).sum((1, 2)), den)
            d = a.diagonal(dim1=-2, dim2=-1)                          # (B, H, N)
            diag[l] = {"cls": d[:, :, 0].mean(1), "patch": d[:, :, 1:].mean((1, 2)),
                       "patch_below_lam": (d[:, :, 1:] < LAM).float().mean((1, 2))}
        h = new
        hd = h[:, 1:] - h[:, 1:].mean(1, keepdim=True)
        pooled[(i + 1, "out")] = {"cls": h[:, 0], "patch": h[:, 1:].mean(1)}
        tok[(i + 1, "in")] = ((hd * ed).sum((1, 2)), ed.pow(2).sum((1, 2)), hd.pow(2).sum((1, 2)))
        tn = F.normalize(h[:, 1:], dim=-1).sum(1)
        n = h.shape[1] - 1
        tok[(i + 1, "cos")] = ((tn.pow(2).sum(-1) - n) / (n * (n - 1)),)   # mean pairwise patch cosine
    return pooled, tok, diag


def collect(net, crops, device, chunk=8):
    N, V = crops.shape[:2]
    acc = {}
    for s in range(0, N, chunk):
        x = crops[s:s + chunk].to(device).flatten(0, 1)
        pooled, tok, diag = terms(net, x)
        for key, d in pooled.items():
            for u, v in d.items():
                acc.setdefault(("pool", key, u), []).append(v.reshape(-1, V, v.shape[-1]).cpu())
        for key, parts in tok.items():
            acc.setdefault(("tok", key), []).append(
                torch.stack(parts, -1).reshape(-1, V, len(parts)).sum(1).cpu())
        for l, d in diag.items():
            for u, v in d.items():
                acc.setdefault(("diag", l, u), []).append(v.reshape(-1, V).mean(1).cpu())
    return {k: torch.cat(v).double() for k, v in acc.items()}


def split_wb(x):
    """(N, V, C) → within (N, V, C) and between (N, C) deviations."""
    c = x.mean(1)
    return x - c[:, None], c - c.mean(0)


def per_image(x, t):
    """Per-image sums for ρ_w, ρ_b and the ratio of h and h + t."""
    xw, xb = split_wb(x)
    tw, tb = split_wb(t)
    yw, yb = xw + tw, xb + tb
    return {"nw": -(tw * xw).sum((1, 2)), "dw": xw.pow(2).sum((1, 2)),
            "nb": -(tb * xb).sum(1), "db": xb.pow(2).sum(1),
            "Wy": yw.pow(2).sum((1, 2)), "By": yb.pow(2).sum(1)}


def log_ratio_change(s, idx):
    f = lambda k: s[k][idx].sum(-1)
    return torch.log(f("Wy") / f("By")) - torch.log(f("dw") / f("db"))


def ratio(x):
    xw, xb = split_wb(x)
    N, V = x.shape[:2]
    return (xw.pow(2).sum() / (N * (V - 1)) / (xb.pow(2).sum() / (N - 1))).item()


def summarize(acc, n_boot=200, seed=0):
    """blocks: B<ℓ> = the terms of block ℓ, against the output of block ℓ−1 they are added to.
    stream: L<ℓ> = the output of block ℓ (L12 = the posthoc_analysis.py L12 unit) and how much
    of it is the input embedding e = patch embedding + position carried through the skip path:
    β = Σ⟨h, e⟩ / Σ‖e‖² (1: e passes verbatim, 0: erased) and share = the squared aggregate
    cosine, for patch-token deviations from their image mean and for the within / between
    parts of the mean patch token (e is the same for every CLS, so CLS has none)."""
    pools = {k[1] for k in acc if k[0] == "pool"}
    N = next(v for k, v in acc.items() if k[0] == "pool").shape[0]
    g = torch.Generator().manual_seed(seed)
    boot = torch.randint(N, (n_boot, N), generator=g)
    full = torch.arange(N)
    blocks = {}
    for l in sorted(l for l, k in pools if k == "ctx"):
        r = {}
        for u in UNITS:
            x = acc[("pool", (l, "h"), u)]
            r[f"ratio_{u}"] = ratio(x)
            for k in TERMS:
                s = per_image(x, acc[("pool", (l, k), u)])
                bt = log_ratio_change(s, boot)
                r[f"{k}_{u}"] = {"rho_w": (s["nw"].sum() / s["dw"].sum()).item(),
                                 "rho_b": (s["nb"].sum() / s["db"].sum()).item(),
                                 "dlogr": log_ratio_change(s, full).item(),
                                 "dlogr_ci": torch.quantile(bt, torch.tensor([.025, .975], dtype=bt.dtype)).tolist()}
        for k in TERMS:
            nd = acc[("tok", (l, k))]
            r[f"{k}_tok"] = (nd[:, 0].sum() / nd[:, 1].sum()).item()
        for u in ("cls", "patch", "patch_below_lam"):
            r[f"diag_{u}"] = acc[("diag", l, u)].mean().item()
        blocks[f"B{l}"] = r
    stream = {}
    ew, eb = split_wb(acc[("pool", (0, "h"), "patch")])
    for l in sorted(l for l, k in pools if k == "out"):
        r = {f"ratio_{u}": ratio(acc[("pool", (l, "out"), u)]) for u in UNITS}
        hw, hb = split_wb(acc[("pool", (l, "out"), "patch")])
        for part, (h, e) in (("w", (hw, ew)), ("b", (hb, eb))):
            he, ee, hh = (h * e).sum(), e.pow(2).sum(), h.pow(2).sum()
            r[f"beta_{part}"] = (he / ee).item()
            r[f"share_{part}"] = (he ** 2 / (ee * hh)).item()
        t = acc[("tok", (l, "in"))].sum(0)                      # Σ⟨h, e⟩, Σ‖e‖², Σ‖h‖²
        r["tok_cos"] = (acc[("tok", (l, "cos"))].sum() / (N * acc[("pool", (l, "out"), "patch")].shape[1])).item()
        r["beta_tok"] = (t[0] / t[1]).item()
        r["share_tok"] = (t[0] ** 2 / (t[1] * t[2])).item()
        stream[f"L{l}"] = r
    return {"blocks": blocks, "stream": stream}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="label of the crop set; all checkpoints must share its data cfg")
    ap.add_argument("arms", nargs="+",
                    help="name=checkpoint, or name=checkpoint@λ to run it with λ_p replaced by λ "
                         "(test-time ablation of the NeuTRENO term)")
    ap.add_argument("--view-images", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="logs/posthoc/terms")
    args = ap.parse_args()
    device = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = False
    arms = dict(a.split("=", 1) for a in args.arms)
    crops = None
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"terms_{args.dataset}.json")
    res = json.load(open(path)) if os.path.exists(path) else {}
    for name, spec in arms.items():
        ckpt, _, lam_test = spec.partition("@")
        ck = load_ckpt(ckpt)
        cfg = ck["cfg"]
        if crops is None:
            crops = seeded_crops(cfg, "validation", args.view_images, args.seed)
        net = build_net(cfg, device)
        net.load_state_dict(ck["net"])
        net.eval()
        if lam_test:
            for blk in net.backbone.blocks:
                blk.attn.lambda_p = float(lam_test)
        mode, lam, alpha = arm_of(net.backbone)
        res[name] = {"path": ckpt, "lambda_test": lam_test or None, "mode": mode, "lambda_p": lam, "alpha": alpha,
                     **summarize(collect(net, crops, device))}
        if mode == "negdiag":
            res[name]["res_lambda"] = {f"B{i + 1}": blk.attn.res_lambda.tolist()
                                       for i, blk in enumerate(net.backbone.blocks) if i > 0}
        print(f"[terms] {name}: {mode} λ={lam} α={alpha}", flush=True)
        del net
        torch.cuda.empty_cache()
        with open(path, "w") as f:                    # after every arm, merged into earlier runs
            json.dump(res, f, indent=1)
    print(f"→ {path}")


if __name__ == "__main__":
    main()
