"""
Export the processed dataset as ordinary image files you can browse.

Mirrors what the original authors' `sort_images.py` did: write every image
into a folder named after its class, so you can flip through them in any
file manager or image viewer.

Output layout:
    data/preview/train/damaged/train_00042.jpg
    data/preview/train/undamaged/train_00043.jpg
    data/preview/val/...
    data/preview/test/...

Filenames carry the split and the row index in the corresponding
`data/processed/X_<split>.npy`, so anything you spot can be traced back.

Also writes contact sheets (grids of many images on one page) to
results/figures/ for a fast overview.

Run:
    python repro/export_images.py                  # all splits
    python repro/export_images.py --split val      # one split
    python repro/export_images.py --limit 200      # first N per class
    python repro/export_images.py --no-sheets
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
PREVIEW = ROOT / "data" / "preview"
FIGS = ROOT / "results" / "figures"

CLASSES = {0: "undamaged", 1: "damaged"}


def export_split(split, limit=None, quality=92):
    X = np.load(DATA / f"X_{split}.npy", mmap_mode="r")
    y = np.load(DATA / f"y_{split}.npy")
    counts = {0: 0, 1: 0}
    for cls, name in CLASSES.items():
        (PREVIEW / split / name).mkdir(parents=True, exist_ok=True)
    for i in range(len(y)):
        cls = int(y[i])
        if limit is not None and counts[cls] >= limit:
            if all(counts[c] >= limit for c in CLASSES):
                break
            continue
        path = PREVIEW / split / CLASSES[cls] / f"{split}_{i:05d}.jpg"
        if not path.exists():
            Image.fromarray(np.asarray(X[i])).save(path, quality=quality)
        counts[cls] += 1
    total = counts[0] + counts[1]
    print(f"  {split}: {total} images "
          f"({counts[1]} damaged, {counts[0]} undamaged) -> "
          f"{PREVIEW / split}")
    return counts


def contact_sheet(split, cls, n=48, cols=8, seed=0):
    X = np.load(DATA / f"X_{split}.npy", mmap_mode="r")
    y = np.load(DATA / f"y_{split}.npy")
    pool = np.flatnonzero(y == cls)
    rng = np.random.default_rng(seed)
    pick = np.sort(rng.choice(pool, min(n, len(pool)), replace=False))
    rows = int(np.ceil(len(pick) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.05))
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, i in zip(np.ravel(axes), pick):
        ax.imshow(np.asarray(X[i]))
        ax.set_title(f"{i}", fontsize=7)
    fig.suptitle(f"{split} / {CLASSES[cls]} - {len(pick)} random samples "
                 f"(titles are row indices in X_{split}.npy)", fontsize=12)
    fig.tight_layout()
    out = FIGS / f"sheet_{split}_{CLASSES[cls]}.png"
    fig.savefig(out, dpi=80)
    plt.close(fig)
    print(f"  wrote {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="all",
                    choices=["train", "val", "test", "all"])
    ap.add_argument("--limit", type=int, default=None,
                    help="max images per class per split")
    ap.add_argument("--no-sheets", action="store_true")
    args = ap.parse_args()

    splits = ["train", "val", "test"] if args.split == "all" else [args.split]
    FIGS.mkdir(parents=True, exist_ok=True)

    print("Exporting individual images:")
    for s in splits:
        export_split(s, limit=args.limit)

    if not args.no_sheets:
        print("Writing contact sheets:")
        for cls in (1, 0):
            contact_sheet("train", cls)

    size = sum(f.stat().st_size for f in PREVIEW.rglob("*.jpg"))
    print(f"\nTotal preview size: {size/2**20:.0f} MB at {PREVIEW}")
    print(f"Browse them in your file manager: {PREVIEW / 'train' / 'damaged'}")


if __name__ == "__main__":
    main()
