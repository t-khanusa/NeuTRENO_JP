"""
Summarize the NeuTRENO × regularizer grid from wandb (project LeJEPA).

For each regularizer prints test accuracy at fixed epochs, the area under the
accuracy curve (mean acc over epochs — the convergence-speed metric), best acc,
and the minimum effective rank of the embedding (transient-collapse indicator),
for lambda_p=0.6 vs lambda_p=0.0 (same network) vs timm.

    python analyze_neutreno_reg.py
    python analyze_neutreno_reg.py --csv logs/neutreno_reg/summary.csv
"""

import argparse
from collections import defaultdict

import wandb

EPOCHS = (10, 25, 50, 75, 100, 200)


def run_rows(project):
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
        g = r.history(keys=["geom/erank_emb", "geom/erank_proj", "geom/token_cos"],
                      samples=10000, pandas=True)
        rows.append({
            "reg": c.get("reg"), "arm": arm, "seed": c.get("seed", 0), "state": r.state,
            "epochs_done": max(acc), "acc": acc,
            "auc": sum(acc.values()) / len(acc),
            "best": max(acc.values()), "final": acc[max(acc)],
            "min_erank_emb": g["geom/erank_emb"].min() if not g.empty else float("nan"),
            "min_erank_proj": g["geom/erank_proj"].min() if not g.empty else float("nan"),
            "token_cos_end": g["geom/token_cos"].iloc[-1] if not g.empty else float("nan"),
        })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="LeJEPA")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = run_rows(args.project)
    by_reg = defaultdict(list)
    for r in rows:
        by_reg[r["reg"]].append(r)

    head = (f"{'reg':11s} {'arm':6s} {'ep':>4s} " + " ".join(f"{'@' + str(e):>6s}" for e in EPOCHS)
            + f" {'AUC':>6s} {'best':>6s} {'erank_h':>7s} {'erank_z':>7s} {'tokcos':>6s}")
    print(head)
    print("-" * len(head))
    for reg in sorted(by_reg):
        for r in sorted(by_reg[reg], key=lambda r: r["arm"], reverse=True):
            accs = " ".join(f"{r['acc'][e]:6.3f}" if e in r["acc"] else f"{'':6s}" for e in EPOCHS)
            print(f"{reg:11s} {r['arm']:6s} {r['epochs_done']:4d} {accs} {r['auc']:6.3f} "
                  f"{r['best']:6.3f} {r['min_erank_emb']:7.2f} {r['min_erank_proj']:7.2f} "
                  f"{r['token_cos_end']:6.3f}{'  (running)' if r['state'] == 'running' else ''}")
        print()

    if args.csv:
        import csv
        with open(args.csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["reg", "arm", "seed", "epoch", "acc"])
            for r in rows:
                for e, a in sorted(r["acc"].items()):
                    w.writerow([r["reg"], r["arm"], r["seed"], e, a])


if __name__ == "__main__":
    main()
