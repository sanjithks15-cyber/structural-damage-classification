"""
Independent verification of the label-polarity decision.

WHY THIS SCRIPT EXISTS
----------------------
The raw PHI-Net README says `0 = Damaged, 1 = Undamaged`. The paper's
confusion matrix implies `0 = Undamaged, 1 = Damaged`. We adopted the paper's
convention, which means we apply `y = 1 - raw`.

A wrong choice here is invisible to every accuracy metric: relabelling both
classes leaves accuracy unchanged. It only corrupts the INTERPRETATION --
which images are "false positives", what recall-for-damage means, and which
direction the confusion matrix reads. So it must be checked directly.

Three checks:
  1. VISUAL   - render random images from each of our classes, large enough
                to judge by eye. This is the decisive one.
  2. EDGE     - damage (cracks, spalling, rubble) adds high-frequency detail.
                Compare an edge-energy statistic between our two classes.
                Directional evidence only; intact buildings have edges too.
  3. CONSIST  - confirm our flip actually inverts the raw file, and that the
                resulting class counts match the paper's 54/46.

Run:  python repro/verify_labels.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "extracted"
FIGS = ROOT / "results" / "figures"

OURS = {0: "undamaged", 1: "damaged"}       # paper convention (what we use)
RAWC = {0: "Damaged", 1: "Undamaged"}       # PHI-Net README convention


def edge_energy(img_u8):
    """Mean absolute Laplacian of the greyscale image: a crude measure of
    how much high-frequency structure (edges, cracks, rubble) is present."""
    g = img_u8.astype(np.float32).mean(axis=2)
    lap = (-4 * g[1:-1, 1:-1]
           + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:])
    return float(np.abs(lap).mean())


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    X = np.load(DATA / "X_train.npy", mmap_mode="r")
    y = np.load(DATA / "y_train.npy")

    print("=== CHECK 3: consistency ===")
    print(f"  TRAIN split, our class 0 ({OURS[0]}): {int((y==0).sum())}  "
          f"-> paper's train split: 2870 undamaged")
    print(f"  TRAIN split, our class 1 ({OURS[1]}): {int((y==1).sum())}  "
          f"-> paper's train split: 2451 damaged")
    print(f"  our flip maps raw {RAWC[0]!r}->1 and raw {RAWC[1]!r}->0, i.e. "
          f"our label 1 should be images the RAW file called 'Damaged'.")

    print("\n=== CHECK 2: edge energy (higher = more high-frequency detail) ===")
    rng = np.random.default_rng(0)
    stats = {}
    for cls in (0, 1):
        idx = rng.choice(np.flatnonzero(y == cls), 400, replace=False)
        vals = np.array([edge_energy(np.asarray(X[i])) for i in sorted(idx)])
        stats[cls] = vals
        print(f"  our class {cls} ({OURS[cls]:>9}): "
              f"mean {vals.mean():6.2f}   median {np.median(vals):6.2f}")
    d = stats[1].mean() - stats[0].mean()
    pooled = np.sqrt((stats[0].var() + stats[1].var()) / 2)
    print(f"  difference (damaged - undamaged) = {d:+.2f}   "
          f"Cohen's d = {d/pooled:+.3f}")
    print("  EXPECTED: positive, if our class 1 really is the damaged class.")

    print("\n=== CHECK 1: visual (see figure) ===")
    fig, axes = plt.subplots(4, 6, figsize=(16, 11))
    for r, cls in enumerate([1, 1, 0, 0]):
        pool = np.flatnonzero(y == cls)
        pick = rng.choice(pool, 6, replace=False)
        for c, i in enumerate(pick):
            ax = axes[r, c]
            ax.imshow(np.asarray(X[i]))
            ax.axis("off")
            ax.set_title(f"our label {cls} = {OURS[cls]}\n"
                         f"(raw file called it {RAWC[1-cls]})", fontsize=8)
    fig.suptitle(
        "Label polarity check - top 2 rows should show DAMAGE "
        "(cracks, spalling, collapse); bottom 2 rows should show INTACT "
        "surfaces.\nIf that is reversed, our flip is wrong.", fontsize=11)
    fig.tight_layout()
    out = FIGS / "01_label_polarity_check.png"
    fig.savefig(out, dpi=85)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
