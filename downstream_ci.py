"""
Paired test-set bootstrap for the linear-probe differences of downstream_eval.py.

Retrains each model's probes on the cached features (same seed, so the metrics match
cls.json), takes the head of the reported lr — best_test or holdout, per model — and
resamples the test images jointly for both models: Δ = B − A with a 95% CI. This is
test-sampling uncertainty of single pre-training runs, not seed-to-seed variance.

    python downstream_ci.py neutreno0.0_visreg_s0_last_ep300 neutreno0.6_visreg_s0_last_ep300
"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np
import torch

from downstream_eval import IN_DOMAIN, LRS, OOD, macro_auroc, train_probe


def per_head(name, m, args, device):
    f = lambda s: torch.load(os.path.join(args.out, "feats", m, f"{name}_{s}.pt"))
    tr, te = f("train"), f("test")
    ml = tr["y"].dim() == 2
    C = tr["y"].shape[1] if ml else int(max(tr["y"].max(), te["y"].max())) + 1
    pa = SimpleNamespace(lrs=LRS, probe_bs=32, probe_epochs=10)
    metric, logits = train_probe(tr["x"], tr["y"], te["x"], te["y"], C, ml, pa, device,
                                 return_logits=True)
    return metric, logits, te["y"], ml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--out", default="logs/downstream")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--boot-auroc", type=int, default=200)
    args = ap.parse_args()
    device = "cuda"
    res = json.load(open(os.path.join(args.out, "cls.json")))
    out = {}
    print(f"Δ = {args.b} − {args.a}, 95% paired bootstrap over test images")
    for name in [d for d in IN_DOMAIN + OOD if d in res[args.a] and d in res[args.b]]:
        out[name] = {}
        heads = {}
        for m in (args.a, args.b):
            metric, logits, y, ml = per_head(name, m, args, device)
            heads[m] = (metric, logits)
        g = np.random.RandomState(0)
        n = len(y)
        for key, lr_key in (("best_test", "best_lr"), ("holdout", "holdout_lr")):
            h = {m: LRS.index(res[m][name][lr_key]) for m in (args.a, args.b)}
            got = {m: heads[m][0][h[m]] for m in h}
            if ml:
                s = {m: torch.sigmoid(heads[m][1][h[m]]).float().numpy() for m in h}
                yy = y.numpy()
                d = []
                for _ in range(args.boot_auroc):
                    i = g.randint(n, size=n)
                    d.append(100 * (macro_auroc(s[args.b][i], yy[i]) - macro_auroc(s[args.a][i], yy[i])))
            else:
                c = {m: (heads[m][1][h[m]].argmax(-1) == y).double().numpy() for m in h}
                diff = c[args.b] - c[args.a]
                i = g.randint(n, size=(args.boot, n))
                d = 100 * diff[i].mean(1)
            lo, hi = np.percentile(d, [2.5, 97.5])
            delta = got[args.b] - got[args.a]
            sig = "+" if lo > 0 else ("−" if hi < 0 else "n.s.")
            out[name][key] = {"a": got[args.a], "b": got[args.b], "delta": delta, "ci": [lo, hi],
                              "verdict": sig,
                              "reproduces_cls_json": all(abs(got[m] - res[m][name][key]) < 0.05 for m in h)}
            print(f"  {name:10s} {key:9s} {got[args.a]:6.2f} → {got[args.b]:6.2f}  "
                  f"Δ={delta:+5.2f} [{lo:+5.2f},{hi:+5.2f}] {sig}"
                  + ("" if out[name][key]["reproduces_cls_json"] else "  (≠ cls.json)"), flush=True)
    with open(os.path.join(args.out, "cls_paired_ci.json"), "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
