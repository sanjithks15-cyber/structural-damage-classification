"""
Classify your own photos as damaged / undamaged, with probabilities.

    .venv/bin/python classify.py photo.jpg
    .venv/bin/python classify.py a.jpg b.png folder_of_photos/
    .venv/bin/python classify.py photo.jpg --show        # open the figure
    .venv/bin/python classify.py photo.jpg --threshold 0.32   # safety-first cutoff

Uses the production model (frozen InceptionV3 + trained head, results/head_full_299.pt).
Writes a figure to results/figures/classify.png.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "repro"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from demo import CLASSES, load_head, open_file, predict_arrays
from inception_transfer import backbone

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def collect(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS)
        elif p.is_file():
            files.append(p)
        else:
            print(f"skipping (not found): {p}")
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("images", nargs="+", help="image files and/or folders")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="call 'damaged' when P(damaged) >= this (default 0.5)")
    ap.add_argument("--show", action="store_true", help="open the figure")
    a = ap.parse_args()

    files = collect(a.images)
    if not files:
        sys.exit("no images found")

    size = 299
    head = load_head(size)
    net = backbone()

    arrs = []
    for f in files:
        img = Image.open(f).convert("RGB").resize((224, 224), Image.BILINEAR)
        arrs.append(np.asarray(img, dtype=np.uint8))
    probs, _ = predict_arrays(np.stack(arrs), head, size, net)

    print(f"\n  threshold: P(damaged) >= {a.threshold:.2f} -> DAMAGED\n")
    print(f"  {'file':<28} {'P(undamaged)':>12} {'P(damaged)':>11}  prediction")
    print("  " + "-" * 66)
    titles = []
    for f, p in zip(files, probs):
        dmg = p[1] >= a.threshold
        label = CLASSES[int(dmg)].upper()
        print(f"  {f.name[:28]:<28} {p[0]:>12.1%} {p[1]:>11.1%}  {label}")
        titles.append((f"{f.name[:22]}\n{label}  P(dam)={p[1]:.0%}",
                       "#b3512f" if dmg else "#2f7d32"))

    cols = min(4, len(files))
    rows = int(np.ceil(len(files) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.6),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, im, t in zip(axes.ravel(), arrs, titles):
        ax.imshow(im)
        ax.set_title(t[0], fontsize=9, color=t[1])
    fig.tight_layout()
    out = ROOT / "results" / "figures" / "classify.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=95)
    print(f"\n  figure: {out}")
    if a.show:
        open_file(out)


if __name__ == "__main__":
    main()
