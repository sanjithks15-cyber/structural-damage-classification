"""
Live demonstration: classify structural images as damaged or undamaged.

Designed to be run in front of an evaluation panel. No training, no network
access, no GPU wait -- it loads the saved classifier head and runs in a couple
of seconds.

    # classify random images from the held-out test set
    python repro/demo.py

    # classify any image file you like
    python repro/demo.py --image /path/to/photo.jpg

    # show the model's mistakes and explain them
    python repro/demo.py --errors

    # overall score on the held-out test set the paper never used
    python repro/demo.py --score

Each run writes a figure to results/figures/demo_*.png and prints a table.
Use --show to pop the figure open.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from inception_transfer import (DEV, IMAGENET_MEAN, IMAGENET_STD, backbone,
                                load_feats)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

CLASSES = ["undamaged", "damaged"]


def load_head(size, repro=False):
    """Load a trained classifier head.

    Default is the production model: trained on all 11,219 available images at
    299px. `repro=True` loads the paper-matched reproduction instead (5,321
    images), for showing the comparison.
    """
    if repro:
        path = RESULTS / f"head_incv3_{size}.pt"
        hint = (f"python repro/inception_transfer.py --extract --train "
                f"--size {size}")
    else:
        path = RESULTS / f"head_full_{size}.pt"
        hint = f"python repro/train_full.py --size {size}"
    if not path.exists():
        raise SystemExit(f"{path} not found.\nTrain it first:\n  {hint}")
    head = nn.Linear(2048, 2).to(DEV)
    head.load_state_dict(torch.load(path, map_location=DEV))
    head.eval()
    return head


@torch.no_grad()
def predict_arrays(imgs_u8, head, size, net=None):
    """imgs_u8: (N,224,224,3) uint8 RGB -> (probs, preds)."""
    net = net or backbone()
    x = torch.from_numpy(np.ascontiguousarray(imgs_u8)).to(DEV)
    x = x.permute(0, 3, 1, 2).float().div_(255.0)
    if x.shape[-1] != size:
        x = F.interpolate(x, size=(size, size), mode="bilinear",
                          align_corners=False)
    x = (x - IMAGENET_MEAN.to(DEV)) / IMAGENET_STD.to(DEV)
    p = torch.softmax(head(net(x)), dim=1).cpu().numpy()
    return p, p.argmax(1)


def show_grid(imgs, titles, suptitle, out, show=False, cols=4):
    rows = int(np.ceil(len(imgs) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.1, rows * 3.5))
    for ax in np.ravel(np.atleast_2d(axes)):
        ax.axis("off")
    for ax, im, t in zip(np.ravel(np.atleast_2d(axes)), imgs, titles):
        ax.imshow(im)
        colour = t[1]
        ax.set_title(t[0], fontsize=9, color=colour)
    fig.suptitle(suptitle, fontsize=13)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=95)
    plt.close(fig)
    print(f"\nfigure: {out}")
    if show:
        subprocess.Popen(["xdg-open", str(out)],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def banner(size, repro=False):
    which = ("reproduction model - 5,321 images, matching the 2018 paper"
             if repro else
             "production model - all 11,219 available images")
    print("=" * 70)
    print("  Structural Damage Classifier")
    print("  InceptionV3 (ImageNet, frozen) + trained FC/softmax head")
    print(f"  {which}")
    print(f"  device: {DEV}   input: {size}x{size}")
    print("=" * 70)


def run_single(path, head, size, show):
    img = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.uint8)[None]
    p, pred = predict_arrays(arr, head, size)
    print(f"\n  file       : {path}")
    print(f"  prediction : {CLASSES[pred[0]].upper()}")
    print(f"  confidence : {p[0].max():.1%}")
    print(f"  (undamaged {p[0][0]:.1%} / damaged {p[0][1]:.1%})")
    colour = "#b3512f" if pred[0] == 1 else "#2f7d32"
    show_grid([arr[0]],
              [(f"{CLASSES[pred[0]]}  p={p[0].max():.2f}", colour)],
              f"Prediction for {Path(path).name}",
              FIGS / "demo_single.png", show, cols=1)


def run_sample(head, size, n, seed, show, only_errors=False):
    X = np.load(DATA / "X_test.npy", mmap_mode="r")
    y = np.load(DATA / "y_test.npy")
    try:
        feats, _ = load_feats("test", size)
    except SystemExit:
        # No cached bottlenecks (fresh clone): compute them now. Takes a
        # couple of seconds for 1460 images and keeps the demo self-contained.
        print("  (no cached features - computing them, ~3s)", flush=True)
        net = backbone()
        feats = np.empty((len(y), 2048), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(y), 32):
                j = min(i + 32, len(y))
                xb = torch.from_numpy(np.ascontiguousarray(X[i:j])).to(DEV)
                xb = xb.permute(0, 3, 1, 2).float().div_(255.0)
                if size != 224:
                    xb = F.interpolate(xb, size=(size, size),
                                       mode="bilinear", align_corners=False)
                xb = (xb - IMAGENET_MEAN.to(DEV)) / IMAGENET_STD.to(DEV)
                feats[i:j] = net(xb).cpu().numpy()
        del net
        torch.cuda.empty_cache()

    with torch.no_grad():
        probs = torch.softmax(
            head(torch.from_numpy(feats).to(DEV)), dim=1).cpu().numpy()
    preds = probs.argmax(1)

    rng = np.random.default_rng(seed)
    if only_errors:
        pool = np.flatnonzero(preds != y)
        pool = pool[np.argsort(-probs[pool].max(1))]   # most confident first
        pick = pool[:n]
        title = ("Where the model fails - held-out test set, "
                 "most confident mistakes first")
    else:
        pick = rng.choice(len(y), n, replace=False)
        title = "Predictions on unseen images - held-out test set"

    imgs, titles = [], []
    print(f"\n  {'image':>10}  {'true':>10}  {'predicted':>10}  {'conf':>6}  ok")
    print("  " + "-" * 50)
    for i in pick:
        ok = preds[i] == y[i]
        imgs.append(np.asarray(X[i]))
        colour = "#2f7d32" if ok else "#b3512f"
        titles.append((f"true: {CLASSES[y[i]]}\n"
                       f"pred: {CLASSES[preds[i]]}  p={probs[i].max():.2f}",
                       colour))
        print(f"  {i:>10}  {CLASSES[y[i]]:>10}  {CLASSES[preds[i]]:>10}  "
              f"{probs[i].max():>6.1%}  {'YES' if ok else 'NO'}")

    acc = (preds == y).mean()
    print(f"\n  accuracy over the whole {len(y)}-image held-out test set: "
          f"{acc:.1%}")
    out = FIGS / ("demo_errors.png" if only_errors else "demo_predictions.png")
    show_grid(imgs, titles, title, out, show)


def run_score(head, size):
    from sklearn import metrics
    feats, y = load_feats("test", size)
    with torch.no_grad():
        pred = head(torch.from_numpy(feats).to(DEV)).argmax(1).cpu().numpy()
    cm = metrics.confusion_matrix(y, pred)
    rep = metrics.classification_report(y, pred, target_names=CLASSES,
                                        digits=3, zero_division=0)
    print(f"\n  Held-out test set: {len(y)} images the model has never seen,")
    print("  and that the original paper never evaluated on at all.\n")
    print(f"  accuracy : {metrics.accuracy_score(y, pred):.1%}")
    print(f"  majority-class baseline : "
          f"{max((y==0).mean(), (y==1).mean()):.1%}\n")
    print("  confusion matrix")
    print(f"                  pred undam   pred dam")
    print(f"    true undam     {cm[0,0]:>8}   {cm[0,1]:>8}")
    print(f"    true dam       {cm[1,0]:>8}   {cm[1,1]:>8}\n")
    print(rep)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image", help="classify a single image file")
    ap.add_argument("--errors", action="store_true",
                    help="show the model's most confident mistakes")
    ap.add_argument("--score", action="store_true",
                    help="full metrics on the held-out test set")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--size", type=int, default=299, choices=[224, 299])
    ap.add_argument("--show", action="store_true",
                    help="open the figure after writing it")
    ap.add_argument("--repro", action="store_true",
                    help="use the paper-matched reproduction model (5321 "
                         "images) instead of the production model")
    args = ap.parse_args()

    banner(args.size, repro=args.repro)
    head = load_head(args.size, repro=args.repro)

    if args.image:
        run_single(args.image, head, args.size, args.show)
    elif args.score:
        run_score(head, args.size)
    else:
        seed = args.seed if args.seed is not None \
            else int(np.random.SeedSequence().entropy % 10000)
        print(f"  random seed: {seed}  (pass --seed {seed} to repeat)")
        run_sample(head, args.size, args.n, seed, args.show,
                   only_errors=args.errors)


if __name__ == "__main__":
    main()
