"""
Summarize the NeuTRENO × regularizer grid from wandb (project LeJEPA).

For each regularizer prints test accuracy at fixed epochs, AUC@K (mean acc over
epochs 1..K — the convergence-speed metric; blank for runs shorter than K, so a
100- and an 800-epoch run are compared on the same window), best acc, and the
minimum participation rank of the embedding (transient-collapse indicator), for
lambda_p=0.6 vs lambda_p=0.0 (same network) vs timm.

Runs logged before the telemetry fix have geom/erank_* (participation ratio with
eigenvalues floored at 1e-8, which reads ≈ d instead of 0 at full collapse); they
are shown with a trailing '*'.

    python analyze_neutreno_reg.py
    python analyze_neutreno_reg.py --auc-at 100 --csv logs/neutreno_reg/summary.csv
"""

import argparse
from collections import defaultdict

import wandb

EPOCHS = (10, 25, 50, 75, 100, 200)


def rank_key(g, rep):
    for k in (f"geom/participation_rank_{rep}", f"geom/erank_{rep}"):
        if k in g and g[k].notna().any():
            return g[k].dropna(), k.startswith("geom/erank")
    return None, False


def run_rows(project, auc_at):
    api = wandb.Api(timeout=60)
    entity = api.default_entity
    rows = []
    for r in api.runs(f"{entity}/{project}"):
        if r.state not in ("finished", "running") or "reg" not in r.config:
            continue
        c = r.config
        bb = c.get("backbone", "neutreno")
        arm = f"lp{float(c.get('lambda_p', 0.0))}" if bb == "neutreno" else "timm"
        h = r.history(keys=["test/acc", "test/epoch"], samples=10000, pandas=True)
        if h.empty:
            continue
        acc = dict(zip(h["test/epoch"].astype(int) + 1, h["test/acc"]))
        # history(keys=...) drops rows missing any key, so read each key alone
        g = {}
        for k in ("geom/participation_rank_emb", "geom/participation_rank_proj",
                  "geom/erank_emb", "geom/erank_proj", "geom/token_cos"):
            h = r.history(keys=[k], samples=10000, pandas=True)
            if not h.empty:
                g[k] = h[k]
        rk_h, old_h = rank_key(g, "emb")
        rk_z, old_z = rank_key(g, "proj")
        done = max(acc)
        rows.append({
            "reg": c.get("reg"), "arm": arm, "seed": c.get("seed", 0), "state": r.state,
            "budget": c.get("epochs"), "epochs_done": done, "acc": acc,
            "auc": (sum(acc[e] for e in range(1, auc_at + 1)) / auc_at
                    if all(e in acc for e in range(1, auc_at + 1)) else float("nan")),
            "best": max(acc.values()), "final": acc[done],
            "min_rank_emb": rk_h.min() if rk_h is not None else float("nan"),
            "min_rank_proj": rk_z.min() if rk_z is not None else float("nan"),
            "old_rank": old_h or old_z,
            "token_cos_end": g["geom/token_cos"].iloc[-1] if "geom/token_cos" in g else float("nan"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="LeJEPA")
    ap.add_argument("--auc-at", type=int, default=100)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = run_rows(args.project, args.auc_at)
    by_reg = defaultdict(list)
    for r in rows:
        by_reg[r["reg"]].append(r)

    head = (f"{'reg':11s} {'arm':6s} {'ep':>4s} " + " ".join(f"{'@' + str(e):>6s}" for e in EPOCHS)
            + f" {'AUC@' + str(args.auc_at):>7s} {'best':>6s} {'prank_h':>7s} {'prank_z':>7s} {'tokcos':>6s}")
    print(head)
    print("-" * len(head))
    for reg in sorted(by_reg):
        for r in sorted(by_reg[reg], key=lambda r: (r["arm"], r["seed"]), reverse=True):
            accs = " ".join(f"{r['acc'][e]:6.3f}" if e in r["acc"] else f"{'':6s}" for e in EPOCHS)
            print(f"{reg:11s} {r['arm']:6s} {r['epochs_done']:4d} {accs} {r['auc']:7.3f} "
                  f"{r['best']:6.3f} {r['min_rank_emb']:7.2f} {r['min_rank_proj']:7.2f} "
                  f"{r['token_cos_end']:6.3f}{'*' if r['old_rank'] else ' '}"
                  f" s{r['seed']}/ep{r['budget']}{'  (running)' if r['state'] == 'running' else ''}")
        print()

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["reg", "arm", "seed", "budget", "epoch", "acc"])
            for r in rows:
                for e, a in sorted(r["acc"].items()):
                    w.writerow([r["reg"], r["arm"], r["seed"], r["budget"], e, a])


if __name__ == "__main__":
    main()
