"""
Step 5b: Confusion matrix and misclassified-image galleries
(report section 5.3, Figs 5).

Produces:
  * confusion matrices for validation AND the held-out PHI-Net test set,
    shown next to the paper's, normalised so they can be compared despite
    different set sizes
  * a gallery of the most confident FALSE POSITIVES (undamaged called
    damaged) and FALSE NEGATIVES (damage missed), captioned with the model's
    probability -- the same presentation as the paper's Fig 5
  * a short text summary of per-class precision/recall

The paper examined these by hand and concluded "patterns can be mistaken as
cracks, and vice versa". The galleries let you check whether the same failure
mode appears here.

Note the paper had NO held-out test set. We report ours separately; it is the
honest number, since the validation set was used while choosing settings.

Run:  python repro/error_analysis.py [--size 224]
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn import metrics

from inception_transfer import DEV, load_feats, train_head

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

CLASSES = ["undamaged", "damaged"]
PAPER_CM = np.array([[448, 86], [117, 349]])


@torch.no_grad()
def probs_for(head, split, size):
    F_, y = load_feats(split, size)
    logits = head(torch.from_numpy(F_).to(DEV))
    p = torch.softmax(logits, dim=1).cpu().numpy()
    return p, y


def draw_cm(ax, cm, title):
    cmn = cm / cm.sum(axis=1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i,j]}\n{cmn[i,j]:.1%}", ha="center",
                    va="center", fontsize=11,
                    color="white" if cmn[i, j] > .5 else "black")
    ax.set_xticks([0, 1], [f"pred\n{c}" for c in CLASSES])
    ax.set_yticks([0, 1], [f"true\n{c}" for c in CLASSES])
    acc = np.trace(cm) / cm.sum()
    ax.set_title(f"{title}\naccuracy {acc:.1%}", fontsize=10)


def gallery(idx, probs, y, split, kind, out, n=8):
    X = np.load(DATA / f"X_{split}.npy", mmap_mode="r")
    src_p = DATA / f"{split}_src_index.npy"
    src = np.load(src_p) if src_p.exists() else None
    if len(idx) == 0:
        print(f"  no {kind} to show")
        return
    pick = idx[:n]
    cols = min(len(pick), 4)
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.5))
    for ax in np.ravel(np.atleast_2d(axes)):
        ax.axis("off")
    for ax, i in zip(np.ravel(np.atleast_2d(axes)), pick):
        ax.imshow(np.asarray(X[i]))
        conf = probs[i].max()
        tag = f"row {i}" + (f" (src {int(src[i])})" if src is not None else "")
        ax.set_title(f"{tag}\ntrue {CLASSES[y[i]]} -> "
                     f"pred {CLASSES[probs[i].argmax()]}  p={conf:.2f}",
                     fontsize=8)
    fig.suptitle(f"{kind} - {split} set, most confident first", fontsize=12)
    fig.tight_layout()
    fig.savefig(out, dpi=90)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)

    print("Training the head (same settings as step 4) ...")
    head, _, _ = train_head(size=args.size, quiet=True)

    out = {}
    cms = {}
    for split in ("val", "test"):
        p, y = probs_for(head, split, args.size)
        pred = p.argmax(1)
        cm = metrics.confusion_matrix(y, pred)
        cms[split] = cm
        rep = metrics.classification_report(
            y, pred, target_names=CLASSES, output_dict=True, zero_division=0)
        acc = metrics.accuracy_score(y, pred)
        out[split] = {
            "accuracy": round(float(acc), 4),
            "confusion": cm.tolist(),
            "damaged_precision": round(rep["damaged"]["precision"], 4),
            "damaged_recall": round(rep["damaged"]["recall"], 4),
            "n": int(len(y)),
        }
        print(f"\n  {split}: acc {acc:.4f}  "
              f"damaged precision {rep['damaged']['precision']:.3f} "
              f"recall {rep['damaged']['recall']:.3f}")
        print(f"       confusion {cm.tolist()}")

        # Most confident mistakes, in the paper's two categories.
        fp = np.flatnonzero((y == 0) & (pred == 1))
        fn = np.flatnonzero((y == 1) & (pred == 0))
        fp = fp[np.argsort(-p[fp, 1])]
        fn = fn[np.argsort(-p[fn, 0])]
        if split == "val":
            gallery(fp, p, y, split, "FALSE POSITIVES (undamaged called damaged)",
                    FIGS / "05_false_positives.png")
            gallery(fn, p, y, split, "FALSE NEGATIVES (damage missed)",
                    FIGS / "06_false_negatives.png")

    # Confusion-matrix comparison figure
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.4))
    draw_cm(axes[0], PAPER_CM, "Paper, report S5.3\n(1000 test images)")
    draw_cm(axes[1], cms["val"], f"Ours, validation\n({cms['val'].sum()} images)")
    draw_cm(axes[2], cms["test"],
            f"Ours, held-out test\n({cms['test'].sum()} images)")
    fig.suptitle("Confusion matrices - reproduction of report S5.3 "
                 "(counts and row-normalised percentages)", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIGS / "04_confusion_matrices.png", dpi=95)
    print(f"\n  wrote {FIGS/'04_confusion_matrices.png'}")

    paper_acc = np.trace(PAPER_CM) / PAPER_CM.sum()
    out["paper_confusion_accuracy"] = round(float(paper_acc), 4)
    out["paper_reported_val_acc"] = 0.830
    (RESULTS / "error_analysis.json").write_text(json.dumps(out, indent=2))
    print(f"  wrote {RESULTS/'error_analysis.json'}")
    print(f"\n  paper's confusion matrix implies {paper_acc:.1%} accuracy "
          f"(its Table 1 claims 83.0%)")


if __name__ == "__main__":
    main()
