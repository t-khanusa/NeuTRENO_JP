"""
Where does the NeuTRENO effect appear, and which "NeuTRENO always ..." claims hold?

Reads the per-sample files of posthoc_analysis.py (<dir>/*.samples.pt + *.json),
pairs every NeuTRENO checkpoint with the same-regularizer, same-kind (best/last)
"only" checkpoint, and tests on paired bootstraps — same val images, same fixed
views, same gradient batches in both arms — with Δ = NeuTRENO − only:

  C1 rank       Δ participation rank, val (pooled) and on per-image view centres,
                at raw CLS / emb / proj and the CLS / mean patch after blocks 3–12
                (participation rank = (tr C)² / ‖C‖_F², no eigendecomposition)
  C2 token sim  Δ final-layer patch-token cosine, residual stream and post-norm
  C3 alignment  Δ cos(∇inv, ∇reg) per block (and Δ|cos|: closer to orthogonal)

A claim "always" survives only if every unit is significant (95% CI) in the
claimed direction. Seeds are not resampled: this is measurement uncertainty of
single runs, not run-to-run variance.

Figures (main = last.pt): per-layer probes, within / between / ratio, token
cosine pre/post, gradient norms and cosine per block.

    python mechanism_claims.py logs/posthoc/mech
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

LAYERS = (3, 6, 9, 12)
RANK_REPS = ["cls", "emb", "proj"] + [f"L{l}_{t}" for l in LAYERS for t in ("cls", "patch")]
BLOCKS = [f"block{i:02d}" for i in range(1, 13)]
REG_COLOR = {"sigreg": "tab:green", "visreg": "tab:blue", "lambdajepa": "tab:orange"}
REG_NAME = {"sigreg": "SIGReg", "visreg": "VISReg", "lambdajepa": "λ-JEPA"}


# ─────────────────────────────────────────────────────────────────────────────
# Loading and pairing
# ─────────────────────────────────────────────────────────────────────────────

def load(d):
    runs = []
    for p in sorted(glob.glob(os.path.join(d, "*.samples.pt"))):
        s = torch.load(p, map_location="cpu", weights_only=False)
        with open(p.replace(".samples.pt", ".json")) as f:
            s["summary"] = json.load(f)
        c = s["cfg"]
        s["reg"] = c["reg"]
        s["neutreno"] = c.get("backbone", "neutreno") == "neutreno" and float(c.get("lambda_p", 0)) > 0
        s["kind"] = os.path.splitext(os.path.basename(s["path"]))[0]
        runs.append(s)
    return runs


def pairs(runs):
    by = defaultdict(dict)
    for r in runs:
        by[(r["reg"], r["kind"])]["neutreno" if r["neutreno"] else "only"] = r
    return {k: v for k, v in sorted(by.items()) if len(v) == 2}


# ─────────────────────────────────────────────────────────────────────────────
# Paired bootstraps
# ─────────────────────────────────────────────────────────────────────────────

def ci(x):
    lo, hi = torch.quantile(x, torch.tensor([0.025, 0.975], dtype=x.dtype)).tolist()
    return lo, hi


def participation_rank(x, idx, device, chunk=25):
    """PR of the covariance of x[idx[b]] for every bootstrap row b → (B,)."""
    x = x.to(device, torch.float64)
    out = []
    for s in range(0, len(idx), chunk):
        z = x[idx[s:s + chunk].to(device)]                         # (b, N, D)
        z = z - z.mean(1, keepdim=True)
        c = z.transpose(1, 2) @ z
        tr = c.diagonal(dim1=1, dim2=2).sum(-1)
        out.append((tr.square() / c.square().sum((1, 2))).cpu())
    return torch.cat(out)


def boot_rank(a, b, B, device, seed=0):
    n = len(a)
    g = torch.Generator().manual_seed(seed)
    idx = torch.cat([torch.arange(n)[None], torch.randint(n, (B, n), generator=g)])
    ra, rb = participation_rank(a, idx, device), participation_rank(b, idx, device)
    d = ra - rb
    return {"neutreno": ra[0].item(), "only": rb[0].item(), "delta": d[0].item(), "ci": ci(d[1:])}


def boot_mean_diff(a, b, B, seed=0):
    """Paired difference of means of per-unit values a, b (n,)."""
    d = (a - b).double()
    g = torch.Generator().manual_seed(seed)
    m = d[torch.randint(len(d), (B, len(d)), generator=g)].mean(1)
    return {"neutreno": a.double().mean().item(), "only": b.double().mean().item(),
            "delta": d.mean().item(), "ci": ci(m)}


def boot_var_split(sa, sb, B, seed=0):
    """Within / between-centre variance and their ratio, paired over images."""
    n = len(sa["within"])
    g = torch.Generator().manual_seed(seed)
    idx = torch.cat([torch.arange(n)[None], torch.randint(n, (B, n), generator=g)])

    def stats(s):
        w = s["within"].double()[idx].mean(1)
        c = s["center"].double()
        bt = torch.stack([c[i].var(0).mean() for i in idx])
        return {"within": w, "between": bt, "ratio": w / bt}

    A, Bs = stats(sa), stats(sb)
    out = {}
    for k in A:
        out[k] = {"neutreno": A[k][0].item(), "only": Bs[k][0].item(),
                  "neutreno_ci": ci(A[k][1:]), "only_ci": ci(Bs[k][1:]),
                  "delta": (A[k][0] - Bs[k][0]).item(), "ci": ci(A[k][1:] - Bs[k][1:])}
    return out


def sign(r, direction):
    """+ significant in the claimed direction, − significant against it, 0 n.s."""
    lo, hi = (direction * v for v in r["ci"])
    return "+" if min(lo, hi) > 0 else ("−" if max(lo, hi) < 0 else "0")


# ─────────────────────────────────────────────────────────────────────────────
# Claims
# ─────────────────────────────────────────────────────────────────────────────

def claims(P, B, device):
    res = {"C1_rank": [], "C2_token_sim": [], "C3_alignment": [], "C3_orthogonality": []}
    for (reg, kind), pr in P.items():
        a, b = pr["neutreno"], pr["only"]
        for rep in RANK_REPS:
            for space, xa, xb in (("val", a["val"][rep], b["val"][rep]),
                                  ("center", a["views"][rep]["center"], b["views"][rep]["center"])):
                r = boot_rank(xa, xb, B, device)
                res["C1_rank"].append({"reg": reg, "kind": kind, "unit": f"{rep}/{space}", **r,
                                       "verdict": sign(r, +1)})
        last = max(a["tokens"])
        for i, where in enumerate(("pre", "post")):
            r = boot_mean_diff(a["tokens"][last][i], b["tokens"][last][i], B * 10)
            res["C2_token_sim"].append({"reg": reg, "kind": kind, "unit": f"L{last}/{where}", **r,
                                        "verdict": sign(r, -1)})
        for blk in BLOCKS + ["embed", "head", "proj"]:
            ca = torch.tensor(a["grads"][blk]["cos"])
            cb = torch.tensor(b["grads"][blk]["cos"])
            r = boot_mean_diff(ca, cb, B * 10)
            res["C3_alignment"].append({"reg": reg, "kind": kind, "unit": blk, **r,
                                        "verdict": sign(r, +1)})
            r = boot_mean_diff(ca.abs(), cb.abs(), B * 10)
            res["C3_orthogonality"].append({"reg": reg, "kind": kind, "unit": blk, **r,
                                            "verdict": sign(r, -1)})
    return res


def print_claims(res):
    title = {"C1_rank": "C1  NeuTRENO always increases rank   (Δ participation rank > 0)",
             "C2_token_sim": "C2  NeuTRENO always lowers final-layer token similarity   (Δ cos < 0)",
             "C3_alignment": "C3  NeuTRENO always aligns ∇inv and ∇reg   (Δ cos(∇inv,∇reg) > 0)",
             "C3_orthogonality": "C3' NeuTRENO makes ∇inv ⟂ ∇reg   (Δ|cos| < 0)"}
    for k, rows in res.items():
        n = defaultdict(int)
        for r in rows:
            n[r["verdict"]] += 1
        verdict = "HOLDS" if n["+"] == len(rows) else ("REFUTED" if n["−"] else "NOT SHOWN")
        print(f"\n## {title[k]}\n   {verdict}: {n['+']} for / {n['0']} n.s. / {n['−']} against "
              f"(of {len(rows)} units)")
        cell = defaultdict(lambda: defaultdict(int))
        for r in rows:
            cell[(r["reg"], r["kind"])][r["verdict"]] += 1
        print("   " + "  ".join(f"{g}/{k}: {c['+']}+ {c['0']}·n.s. {c['−']}−" for (g, k), c in cell.items()))
        print(f"   {'reg':11s} {'kind':5s} {'unit':16s} {'NeuTRENO':>9s} {'only':>9s} "
              f"{'Δ':>8s} {'95% CI':>19s}")
        for r in rows:
            print(f"   {r['reg']:11s} {r['kind']:5s} {r['unit']:16s} {r['neutreno']:9.4g} "
                  f"{r['only']:9.4g} {r['delta']:+8.3g} [{r['ci'][0]:+8.3g},{r['ci'][1]:+8.3g}] "
                  f"{r['verdict']}")


# ─────────────────────────────────────────────────────────────────────────────
# Figures
# ─────────────────────────────────────────────────────────────────────────────

def _style(reg, neutreno):
    return dict(color=REG_COLOR.get(reg, "k"), ls="-" if neutreno else "--",
                marker="o" if neutreno else "s", ms=4,
                label=f"{REG_NAME.get(reg, reg)} {'NeuTRENO' if neutreno else 'only'}")


def fig_probes(P, kind, path):
    fig, ax = plt.subplots(2, 2, figsize=(9, 7), sharex=True)
    for (reg, k), pr in P.items():
        if k != kind:
            continue
        for arm, r in pr.items():
            pb = r["summary"]["probe"]
            for i, tok in enumerate(("cls", "patch")):
                for j, m in enumerate(("linear", "knn")):
                    ax[i, j].plot(LAYERS, [100 * pb[f"L{l}_{tok}"][m] for l in LAYERS],
                                  **_style(reg, arm == "neutreno"))
    for i, tok in enumerate(("CLS", "patch-mean")):
        for j, m in enumerate(("linear probe", "k-NN (k=20)")):
            ax[i, j].set_title(f"{tok} — {m}")
            ax[i, j].set_ylabel("val acc (%)")
            ax[i, j].grid(alpha=.3)
    for a in ax[1]:
        a.set_xlabel("block")
        a.set_xticks(LAYERS)
    ax[0, 0].legend(fontsize=8)
    fig.suptitle(f"Frozen probes per layer ({kind}.pt)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_variance(P, kind, B, path):
    fig, ax = plt.subplots(2, 3, figsize=(13, 7), sharex=True)
    for (reg, k), pr in P.items():
        if k != kind:
            continue
        a, b = pr["neutreno"], pr["only"]
        for i, tok in enumerate(("cls", "patch")):
            st = [boot_var_split(a["views"][f"L{l}_{tok}"], b["views"][f"L{l}_{tok}"], B)
                  for l in LAYERS]
            for j, q in enumerate(("within", "between", "ratio")):
                for arm in ("neutreno", "only"):
                    y = [s[q][arm] for s in st]
                    lo = [y_ - s[q][f"{arm}_ci"][0] for y_, s in zip(y, st)]
                    hi = [s[q][f"{arm}_ci"][1] - y_ for y_, s in zip(y, st)]
                    ax[i, j].errorbar(LAYERS, y, yerr=[lo, hi], capsize=3,
                                      **_style(reg, arm == "neutreno"))
    names = {"within": "within-image variance (across views)",
             "between": "between-image variance (view centres)",
             "ratio": "within / between"}
    for i, tok in enumerate(("CLS", "patch-mean")):
        for j, q in enumerate(("within", "between", "ratio")):
            ax[i, j].set_title(f"{tok} — {names[q]}", fontsize=10)
            ax[i, j].grid(alpha=.3)
            if q != "ratio":
                ax[i, j].set_yscale("log")
    for a_ in ax[1]:
        a_.set_xlabel("block")
        a_.set_xticks(LAYERS)
    ax[0, 2].legend(fontsize=8)
    fig.suptitle(f"Variance split per layer, 512 val images × 8 fixed views ({kind}.pt); "
                 "absolute variances are in each model's residual-stream scale, the ratio is scale-free",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_tokens_grads(P, kind, path):
    fig, ax = plt.subplots(1, 5, figsize=(22, 4.2))
    for (reg, k), pr in P.items():
        if k != kind:
            continue
        for arm, r in pr.items():
            st = _style(reg, arm == "neutreno")
            ls = sorted(r["tokens"])
            for i in range(2):
                ax[i].plot(ls, [r["tokens"][l][i].double().mean().item() for l in ls], **st)
            g = r["grads"]
            x = range(1, 13)
            for j, q in enumerate(("norm_inv", "norm_reg")):
                ax[2 + j].plot(x, [sum(g[b][q]) / len(g[b][q]) for b in BLOCKS], **st)
            c = [torch.tensor(g[b]["cos"]) for b in BLOCKS]
            ax[4].errorbar(x, [v.mean().item() for v in c],
                           yerr=[v.std().item() / len(v) ** .5 for v in c], capsize=2, **st)
    for a_, t in zip(ax, ("patch-token cos, residual stream", "patch-token cos, post-LayerNorm",
                          "‖∇ weighted invariance‖", "‖∇ weighted regularizer‖",
                          "cos(∇inv, ∇reg)  (±s.e. over batches)")):
        a_.set_title(t, fontsize=10)
        a_.set_xlabel("block")
        a_.grid(alpha=.3)
    ax[2].set_yscale("log")
    ax[3].set_yscale("log")
    ax[4].axhline(0, color="k", lw=.8)
    ax[0].legend(fontsize=8)
    fig.suptitle(f"Token similarity and per-term gradients per block ({kind}.pt)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    P = pairs(load(args.dir))
    print("pairs:", ", ".join(f"{r}/{k}" for r, k in P))
    res = claims(P, args.boot, args.device)
    print_claims(res)
    with open(os.path.join(args.dir, "claims.json"), "w") as f:
        json.dump(res, f, indent=1)

    figs = os.path.join(args.dir, "figs")
    os.makedirs(figs, exist_ok=True)
    for kind in sorted({k for _, k in P}):
        fig_probes(P, kind, os.path.join(figs, f"probes_layers_{kind}.png"))
        fig_variance(P, kind, args.boot, os.path.join(figs, f"variance_layers_{kind}.png"))
        fig_tokens_grads(P, kind, os.path.join(figs, f"tokens_grads_{kind}.png"))
    print(f"\nfigures → {figs}")


if __name__ == "__main__":
    main()
