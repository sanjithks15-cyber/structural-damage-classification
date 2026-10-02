"""
Step 5a: Learning curve / bias-variance diagnostic (report section 5.2, Fig 4).

The paper trained the frozen-InceptionV3 + FC model at a range of dataset
sizes and observed that training and validation accuracy sit close together
and both plateau low -- the signature of a BIAS-dominated model. That reading
is what motivated everything in their section 5.4 (unfreeze more layers to add
capacity, then fight the resulting variance).

We reproduce the diagnostic and read it ourselves, rather than assuming their
conclusion transfers: our frozen model shows a wider train/val gap than
theirs, so the bias/variance balance may genuinely differ.

Protocol
--------
  * subsample the 5321-image training set to n images, class ratio preserved
  * train the identical FC+softmax head (GD, lr 0.01, batch 100, 4000 iters)
  * evaluate on the FIXED 592-image validation set
  * repeat over `--seeds` seeds and plot mean +/- spread

Holding the validation set fixed differs from the paper, which appears to have
scaled a 90/10 split together. A fixed validation set is less noisy and makes
the points comparable; with only 592 images the alternative would add several
points of jitter at the small end.

Run:  python repro/learning_curve.py [--seeds 3]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

from inception_transfer import train_head

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

SIZES = [200, 400, 800, 1200, 1800, 2500, 3200, 4000, 4700, 5321]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()

    rows = []
    for n in SIZES:
        tr_accs, va_accs = [], []
        for s in range(args.seeds):
            _, _, (ytr, tr_pred, yva, va_pred) = train_head(
                size=args.size, n_train=n, seed=229 + s, quiet=True)
            tr_accs.append(metrics.accuracy_score(ytr, tr_pred))
            va_accs.append(metrics.accuracy_score(yva, va_pred))
        row = {
            "n_train": int(n),
            "n_train_actual": int(len(ytr)),
            "train_acc_mean": float(np.mean(tr_accs)),
            "train_acc_std": float(np.std(tr_accs)),
            "val_acc_mean": float(np.mean(va_accs)),
            "val_acc_std": float(np.std(va_accs)),
            "gap": float(np.mean(tr_accs) - np.mean(va_accs)),
        }
        rows.append(row)
        print(f"  n={n:>5} (actual {row['n_train_actual']:>5})  "
              f"train {row['train_acc_mean']:.3f}+-{row['train_acc_std']:.3f}  "
              f"val {row['val_acc_mean']:.3f}+-{row['val_acc_std']:.3f}  "
              f"gap {row['gap']:+.3f}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "learning_curve.json").write_text(json.dumps(rows, indent=2))

    ns = [r["n_train_actual"] for r in rows]
    tr = np.array([r["train_acc_mean"] for r in rows])
    trs = np.array([r["train_acc_std"] for r in rows])
    va = np.array([r["val_acc_mean"] for r in rows])
    vas = np.array([r["val_acc_std"] for r in rows])

    FIGS.mkdir(parents=True, exist_ok=True)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    ax.plot(ns, tr, "o-", label="training accuracy", color="#b3512f")
    ax.fill_between(ns, tr - trs, tr + trs, alpha=.18, color="#b3512f")
    ax.plot(ns, va, "s-", label="validation accuracy", color="#2f6fb3")
    ax.fill_between(ns, va - vas, va + vas, alpha=.18, color="#2f6fb3")
    ax.axhline(0.534, ls=":", c="grey", label="majority-class baseline (53.4%)")
    ax.axhline(0.830, ls="--", c="green", lw=1,
               label="paper's reported val acc (83.0%)")
    ax.set_xlabel("number of training images")
    ax.set_ylabel("accuracy")
    ax.set_title("Learning curve - frozen InceptionV3 + FC/softmax")
    ax.set_ylim(0.45, 1.0)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=.25)

    ax2.plot(ns, tr - va, "d-", color="#6a3fb3")
    ax2.axhline(0, c="grey", lw=.8)
    ax2.set_xlabel("number of training images")
    ax2.set_ylabel("train acc - val acc")
    ax2.set_title("Generalisation gap (variance indicator)")
    ax2.grid(alpha=.25)

    fig.suptitle("Bias/variance diagnostic - reproduction of report Fig. 4",
                 fontsize=12)
    fig.tight_layout()
    out = FIGS / "03_learning_curve.png"
    fig.savefig(out, dpi=95)
    print(f"\nwrote {out}")
    print(f"wrote {RESULTS/'learning_curve.json'}")


if __name__ == "__main__":
    main()
