"""
BEYOND THE PAPER (2): choosing an operating point for triage.

The paper reports accuracy and nothing else. For the application it describes
-- prioritising which buildings an engineer inspects after an earthquake --
accuracy is the wrong figure of merit, because the two errors are not equally
costly:

    false positive  an intact building gets inspected      wasted trip
    false negative  a damaged building is cleared          someone walks into it

Both models in this repository use the default decision rule, argmax, which is
exactly "damaged if P(damaged) > 0.5". Nobody chose 0.5; it is simply what
argmax does. This script asks what it costs us and what a deliberate choice
would look like.

It produces:
  * a precision-recall curve and a threshold sweep on the held-out test set
  * ROC AUC, which is threshold-independent and so measures the ranking
    quality of the model rather than one arbitrary cut
  * the threshold that reaches a target recall ON VALIDATION, with the
    resulting test metrics -- choosing on validation and reporting on test,
    because picking the threshold on test would be cheating
  * a reliability diagram: when the model says 80%, is it right 80% of the
    time? An uncalibrated model can rank well and still lie about confidence.

Additive only: no existing file or result is modified.

Run:  python repro/operating_point.py
      python repro/operating_point.py --target-recall 0.90
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
FEAT = ROOT / "results" / "features"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def damaged_prob(split, size=299):
    """P(damaged) from the production model, for one split."""
    head = nn.Linear(2048, 2).to(DEV)
    head.load_state_dict(torch.load(RESULTS / f"head_full_{size}.pt",
                                    map_location=DEV))
    head.eval()
    tag = "" if size == 224 else f"_{size}"
    f = np.load(FEAT / f"incv3_{split}{tag}.npy")
    y = np.load(DATA / f"y_{split}.npy")
    with torch.no_grad():
        p = torch.softmax(head(torch.from_numpy(f).to(DEV)), 1).cpu().numpy()
    return p[:, 1], y          # probability of class 1 = damaged


def metrics_at(p, y, thr):
    pred = (p >= thr).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "threshold": float(thr),
        "accuracy": float((tp + tn) / len(y)),
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "missed_damaged": int(fn),
        "false_alarms": int(fp),
        "confusion": cm.tolist(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-recall", type=float, default=0.90)
    ap.add_argument("--size", type=int, default=299)
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)

    pv, yv = damaged_prob("val", args.size)
    pt, yt = damaged_prob("test", args.size)

    auc = roc_auc_score(yt, pt)
    print(f"  ROC AUC on the held-out test set : {auc:.4f}")
    print("  (threshold-independent: how well the model RANKS damaged above "
          "undamaged)\n")

    base = metrics_at(pt, yt, 0.5)
    print("  At the default threshold of 0.5 (what argmax does):")
    print(f"    accuracy {base['accuracy']:.1%}   recall {base['recall']:.1%}"
          f"   precision {base['precision']:.1%}")
    print(f"    misses {base['missed_damaged']} damaged buildings, "
          f"{base['false_alarms']} false alarms\n")

    # choose a threshold on VALIDATION that reaches the target recall
    order = np.argsort(-pv)
    tgt = None
    for thr in np.unique(np.round(pv, 4))[::-1]:
        if metrics_at(pv, yv, thr)["recall"] >= args.target_recall:
            tgt = float(thr)
            break
    chosen = metrics_at(pt, yt, tgt) if tgt is not None else None
    if chosen:
        print(f"  Threshold {tgt:.3f}, chosen on VALIDATION to reach "
              f"{args.target_recall:.0%} recall.")
        print(f"  Its performance on the untouched test set:")
        print(f"    accuracy {chosen['accuracy']:.1%}   "
              f"recall {chosen['recall']:.1%}   "
              f"precision {chosen['precision']:.1%}")
        print(f"    misses {chosen['missed_damaged']} damaged buildings "
              f"(was {base['missed_damaged']}), "
              f"{chosen['false_alarms']} false alarms "
              f"(was {base['false_alarms']})")
        d_missed = base["missed_damaged"] - chosen["missed_damaged"]
        d_alarms = chosen["false_alarms"] - base["false_alarms"]
        if d_missed > 0:
            print(f"    -> catches {d_missed} more damaged buildings at the "
                  f"cost of {d_alarms} extra inspections "
                  f"({d_alarms/max(d_missed,1):.1f} wasted trips per building "
                  f"saved)")

    prec, rec, thr_pr = precision_recall_curve(yt, pt)
    fpr, tpr, _ = roc_curve(yt, pt)
    sweep = [metrics_at(pt, yt, t) for t in np.linspace(0.05, 0.95, 19)]

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))

    ax[0].plot(rec, prec, lw=2, color="#17365d")
    ax[0].scatter([base["recall"]], [base["precision"]], color="#b3512f",
                  zorder=5, label="default threshold 0.5")
    if chosen:
        ax[0].scatter([chosen["recall"]], [chosen["precision"]],
                      color="#2f7d32", zorder=5,
                      label=f"chosen on val (thr={tgt:.2f})")
    ax[0].set_xlabel("recall - fraction of damaged buildings caught")
    ax[0].set_ylabel("precision")
    ax[0].set_title("Precision-recall trade-off")
    ax[0].grid(alpha=.3)
    ax[0].legend(fontsize=8)

    ts = [m["threshold"] for m in sweep]
    ax[1].plot(ts, [m["recall"] for m in sweep], "o-", label="recall (damaged)")
    ax[1].plot(ts, [m["precision"] for m in sweep], "s-", label="precision")
    ax[1].plot(ts, [m["accuracy"] for m in sweep], "^-", label="accuracy")
    ax[1].axvline(0.5, ls=":", c="grey", label="default")
    ax[1].set_xlabel("decision threshold")
    ax[1].set_title("What the threshold controls")
    ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=8)

    # reliability diagram
    bins = np.linspace(0, 1, 11)
    idx = np.digitize(pt, bins) - 1
    xs, ys, ns = [], [], []
    for b in range(10):
        m = idx == b
        if m.sum() >= 10:
            xs.append(pt[m].mean())
            ys.append(yt[m].mean())
            ns.append(int(m.sum()))
    ax[2].plot([0, 1], [0, 1], ls="--", c="grey", label="perfect calibration")
    ax[2].plot(xs, ys, "o-", color="#6a3fb3", label="our model")
    ax[2].set_xlabel("predicted P(damaged)")
    ax[2].set_ylabel("actual fraction damaged")
    ax[2].set_title("Reliability - is 80% confidence really 80%?")
    ax[2].grid(alpha=.3)
    ax[2].legend(fontsize=8)

    fig.suptitle("Operating point analysis - held-out test set "
                 f"(ROC AUC {auc:.3f})", fontsize=13)
    fig.tight_layout()
    out = FIGS / "07_operating_point.png"
    fig.savefig(out, dpi=95)
    print(f"\n  wrote {out}")

    ece = float(np.average(np.abs(np.array(xs) - np.array(ys)),
                           weights=ns)) if xs else float("nan")
    print(f"  expected calibration error: {ece:.3f} "
          f"(0 = perfectly calibrated)")

    payload = {"roc_auc": float(auc), "default": base,
               "target_recall": args.target_recall,
               "chosen_threshold": tgt, "at_chosen": chosen,
               "sweep": sweep, "calibration_error": ece}
    (RESULTS / "operating_point.json").write_text(json.dumps(payload, indent=2))
    print(f"  wrote {RESULTS/'operating_point.json'}")


if __name__ == "__main__":
    main()
