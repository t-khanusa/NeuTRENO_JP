"""
Small successive-halving search for training_neutreno_reg.py at a fixed budget.

Every config trains on the full `--epochs` cosine schedule; rungs only decide how
far a config gets before the next cut, so a survivor's final run is exactly the
run it would have been without the search (stop_after + resume=auto, full state).

Default grid (one-at-a-time around the current default lr=2e-3, visreg_lamb=0.6):
    lr ∈ {1e-3, 2e-3, 4e-3} at visreg_lamb=0.6,   visreg_lamb ∈ {0.4, 0.8} at lr=2e-3
Rungs 50 → 150 → 300 keep 5 → 2 → 1 (≈ 600 epochs for one lambda_p).

Score at a rung = mean online-probe val accuracy over its last `--window` epochs.
ImageNette has no separate val split, so the pick is selected on the reporting
split — report the chosen config with a fresh seed if it goes in a table.

    python hp_halving.py --lambda-p 0.0 --gpu 0
    python hp_halving.py --lambda-p 0.6 --gpu 1
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime

import torch

PY = sys.executable
GRID = [(1e-3, 0.6), (2e-3, 0.6), (4e-3, 0.6), (2e-3, 0.4), (2e-3, 0.8)]


def run_name(lp, lr, vl):
    return f"neutreno{lp}_visreg_lr{lr:g}_vl{vl:g}"


def last_state(ckpt_root, name, epochs):
    path = os.path.join(ckpt_root, f"{name}_imagenette_s0_ep{epochs}", "last.pt")
    if not os.path.exists(path):
        return 0, []
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return ck["epoch"] + 1, list(ck.get("acc_hist", []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lambda-p", type=float, required=True)
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--rungs", type=int, nargs="+", default=[50, 150, 300])
    ap.add_argument("--keep", type=int, nargs="+", default=[2, 1],
                    help="configs kept after each rung but the last")
    ap.add_argument("--window", type=int, default=3)
    ap.add_argument("--ckpt-root", default="checkpoints")
    ap.add_argument("--log-dir", default="logs/hp_visreg")
    ap.add_argument("--extra", nargs="*", default=[], help="extra hydra overrides (smoke tests)")
    args = ap.parse_args()
    assert args.rungs[-1] == args.epochs and len(args.keep) == len(args.rungs) - 1

    os.makedirs(args.log_dir, exist_ok=True)
    lp = args.lambda_p
    state_path = os.path.join(args.log_dir, f"halving_lp{lp}.json")
    alive = [run_name(lp, lr, vl) for lr, vl in GRID]
    hp = {run_name(lp, lr, vl): {"lr": lr, "visreg_lamb": vl} for lr, vl in GRID}
    history = []
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(args.gpu),
           "HF_DATASETS_OFFLINE": "1", "HF_HUB_OFFLINE": "1"}

    for ri, rung in enumerate(args.rungs):
        scores = {}
        for name in alive:
            done, _ = last_state(args.ckpt_root, name, args.epochs)
            if done < rung:
                cmd = [PY, "training_neutreno_reg.py", "reg=visreg", "backbone=neutreno",
                       f"lambda_p={lp}", f"lr={hp[name]['lr']}",
                       f"visreg_lamb={hp[name]['visreg_lamb']}", f"epochs={args.epochs}",
                       f"stop_after={rung}", f"run_name={name}", "resume=auto",
                       f"ckpt_root={args.ckpt_root}", *args.extra]
                print(f"=== [gpu{args.gpu}] {name} → ep{rung} {datetime.now():%F %T}", flush=True)
                with open(os.path.join(args.log_dir, f"{name}.log"), "a") as f:
                    rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=env).returncode
                if rc:
                    print(f"!!! {name} failed (rc={rc}), see {args.log_dir}/{name}.log", flush=True)
            done, accs = last_state(args.ckpt_root, name, args.epochs)
            win = accs[max(rung - args.window, 0):rung]
            scores[name] = sum(win) / len(win) if done >= rung and win else float("-inf")
        ranked = sorted(alive, key=lambda n: scores[n], reverse=True)
        history.append({"rung": rung, "scores": {n: scores[n] for n in ranked}})
        print(f"--- rung ep{rung}: " + ", ".join(f"{n}={100 * scores[n]:.2f}" for n in ranked),
              flush=True)
        if ri < len(args.keep):
            alive = ranked[:args.keep[ri]]
        with open(state_path, "w") as f:
            json.dump({"lambda_p": lp, "grid": hp, "rungs": history,
                       "alive": alive, "best": ranked[0]}, f, indent=1)

    print(f"=== [gpu{args.gpu}] lambda_p={lp} best: {ranked[0]} {hp[ranked[0]]} "
          f"acc@{args.window}={100 * scores[ranked[0]]:.2f}  {datetime.now():%F %T}", flush=True)


if __name__ == "__main__":
    main()
