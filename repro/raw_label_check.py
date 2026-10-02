"""
Show images annotated with the RAW dataset label, verbatim, for human review.

This deliberately bypasses our interpretation. It reads `task2_y_train.npy`
straight from the PHI-Net distribution and prints the class name exactly as
the dataset's own README defines it:

    PHI-Net README:  0: Damaged state (D)      1: Undamaged state (UD)

If the README is right, every image titled "Damaged" shows damage. If the
images disagree with the titles, the README is wrong and our flip
(y = 1 - y_raw) is correct.

Side effect: writes the source-index mapping that `prepare_data.py` should
have saved, so any processed row can be traced back to its row in the
original distribution.

Run:  python repro/raw_label_check.py [--n 16] [--seed 0]
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw" / "extracted"
FIGS = ROOT / "results" / "figures"

# Verbatim from the README inside phi_task2_1.zip:
README_NAMES = {0: "Damaged (D)", 1: "Undamaged (UD)"}

# Must match prepare_data.py exactly.
SEED = 229
N_DAMAGED, N_UNDAMAGED = 2727, 3186
N_VAL_DAMAGED, N_VAL_UNDAMAGED = 276, 316


def stratified_take(labels, n_per_class, rng):
    picked = []
    for cls, n in n_per_class.items():
        pool = np.flatnonzero(labels == cls)
        picked.append(rng.choice(pool, size=n, replace=False))
    idx = np.concatenate(picked)
    rng.shuffle(idx)
    return idx


def recover_source_indices():
    """Replay prepare_data.py's RNG to recover which original rows we kept."""
    y_raw_onehot = np.load(RAW / "task2_y_train.npy")
    y_pool = 1 - np.argmax(y_raw_onehot, axis=1)   # our (flipped) convention
    rng = np.random.default_rng(SEED)
    sel = stratified_take(y_pool, {0: N_UNDAMAGED, 1: N_DAMAGED}, rng)
    y_sel = y_pool[sel]
    val_local = stratified_take(
        y_sel, {0: N_VAL_UNDAMAGED, 1: N_VAL_DAMAGED}, rng
    )
    mask = np.zeros(len(sel), dtype=bool)
    mask[val_local] = True
    return np.sort(sel[~mask]), np.sort(sel[mask]), y_raw_onehot


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    FIGS.mkdir(parents=True, exist_ok=True)

    tr_src, va_src, y_raw_onehot = recover_source_indices()

    # Self-check: the recovered indices must reproduce the saved labels.
    y_train_saved = np.load(DATA / "y_train.npy")
    y_from_src = 1 - np.argmax(y_raw_onehot[tr_src], axis=1)
    ok = np.array_equal(y_train_saved, y_from_src)
    print(f"index recovery self-check: {'PASS' if ok else 'FAIL'} "
          f"({len(tr_src)} train, {len(va_src)} val rows recovered)")
    if not ok:
        raise SystemExit("Recovered indices do not match the saved split.")

    np.save(DATA / "train_src_index.npy", tr_src)
    np.save(DATA / "val_src_index.npy", va_src)
    print(f"wrote {DATA/'train_src_index.npy'} and val_src_index.npy")

    # Build the blind-ish review grid: titles come ONLY from the raw file.
    X = np.load(DATA / "X_train.npy", mmap_mode="r")
    rng = np.random.default_rng(args.seed)
    rows = rng.choice(len(tr_src), args.n, replace=False)

    cols = 4
    nrows = int(np.ceil(args.n / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 3.3, nrows * 3.6))
    for ax in np.ravel(axes):
        ax.axis("off")
    print("\n  processed_row  source_row  raw_onehot   README says")
    for ax, r in zip(np.ravel(axes), rows):
        src = int(tr_src[r])
        onehot = y_raw_onehot[src]
        raw_cls = int(np.argmax(onehot))
        ax.imshow(np.asarray(X[r]))
        ax.set_title(f"source row {src}\n"
                     f"y_raw = {onehot.astype(int).tolist()}  ->  "
                     f"{README_NAMES[raw_cls]}", fontsize=9)
        print(f"  {r:>13}  {src:>10}  {onehot.astype(int).tolist()}   "
              f"{README_NAMES[raw_cls]}")

    fig.suptitle(
        "Labels taken VERBATIM from task2_y_train.npy, named per the PHI-Net "
        "README (0 = Damaged, 1 = Undamaged).\n"
        "Question: does each title match its image?", fontsize=12)
    fig.tight_layout()
    out = FIGS / "02_raw_label_review.png"
    fig.savefig(out, dpi=85)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
