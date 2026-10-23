"""
Which part of the NeuTRENO residual carries its effect? Compares the residual-term
intervention arms (training_neutreno.py RES_MODES) against a common baseline on the
per-sample files of posthoc_analysis.py, with the paired bootstraps of
mechanism_claims.py (same val images, same fixed views):

  full    Attn + λ(V0 − V)      anchor  Attn + λV0      sub   Attn − λV
  scale   x + α·AttnBranch(x)   lp0     NeuTRENO-ViT at λ = 0   timm  plain ViT
  negdiag (A − λ_h I)V, λ_h learned per head   corner  ((1−λ)A + λI)V

Effects (the ones that survived mechanism_claims.py, plus rank and accuracy):
  ratio   within / between-image variance of the view centres, L9 / L12 CLS and patch
  tok     residual-stream patch-token cosine, mean over blocks 2–11
  rank    participation rank of the val patch tokens after blocks 6 and 9
  probe   frozen linear probe on the raw CLS (accuracy, no CI: one fit per model)

    python intervention_effects.py logs/posthoc/mech --reg visreg --kind last --base timm
    python intervention_effects.py logs/posthoc/galaxy10 --base lp0 --epoch 400
"""

import argparse
import json
import os

import torch

from mechanism_claims import boot_mean_diff, boot_rank, boot_var_split, load, sign

ARMS = ["timm", "lp0", "full", "anchor", "sub", "scale0.4", "negdiag", "corner"]
RATIO_UNITS = ["L9_cls", "L9_patch", "L12_cls", "L12_patch"]
RANK_UNITS = ["L6_patch", "L9_patch"]


def mid_token_cos(r):
    return torch.stack([r["tokens"][b][0] for b in range(2, 12)]).double().mean(0)


def compare(a, b, B, device):
    rows = []
    for u in RATIO_UNITS:
        v = boot_var_split(a["views"][u], b["views"][u], B)["ratio"]
        rows.append((f"ratio {u}", v, sign(v, -1)))
    v = boot_mean_diff(mid_token_cos(a), mid_token_cos(b), B * 10)
    rows.append(("tok cos L2–11", v, sign(v, -1)))
    for u in RANK_UNITS:
        v = boot_rank(a["val"][u], b["val"][u], B, device)
        rows.append((f"rank {u}", v, sign(v, +1)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--reg", default="visreg")
    ap.add_argument("--kind", default="last")
    ap.add_argument("--base", default="lp0", choices=ARMS)
    ap.add_argument("--epoch", type=int, default=100)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    runs = {r["arm"]: r for r in load(args.dir)
            if r["reg"] == args.reg and r["kind"] == args.kind and r["summary"]["epoch"] == args.epoch}
    print("arms:", ", ".join(f"{k}={runs[k]['name']}" for k in ARMS if k in runs))
    base = runs[args.base]
    out = {}
    print(f"\nΔ = arm − {args.base}; verdict + / − / 0 = significant in the NeuTRENO (full) "
          f"direction / against it / n.s. (95% paired bootstrap)")
    print(f"{'arm':9s} {'probe cls':>9s}  " + "  ".join(f"{u:>26s}" for u, *_ in compare(base, base, 2, args.device)))
    for k in ARMS:
        if k not in runs or k == args.base:
            continue
        rows = compare(runs[k], base, args.boot, args.device)
        acc = 100 * runs[k]["summary"]["probe"]["cls"]["linear"]
        out[k] = {"probe_cls": acc, **{u: {**v, "verdict": s} for u, v, s in rows}}
        print(f"{k:9s} {acc:9.2f}  " + "  ".join(
            f"{v['delta']:+8.3g} [{v['ci'][0]:+.2g},{v['ci'][1]:+.2g}] {s}" for _, v, s in rows))
    acc = 100 * base["summary"]["probe"]["cls"]["linear"]
    print(f"{args.base:9s} {acc:9.2f}  " + "  ".join(
        f"{'(' + format(v['only'], '.4g') + ')':>26s}" for _, v, _ in compare(base, base, 2, args.device)))
    path = os.path.join(args.dir, f"interventions_{args.reg}_{args.kind}_vs_{args.base}.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\n→ {path}")


if __name__ == "__main__":
    main()
