"""
Live demonstration of the final model.

The final model is an ensemble: three frozen modern backbones
(EfficientNetV2, ConvNeXt, Swin), each with its own RBF-kernel SVM, with the
three probability outputs averaged. 88.5% on the held-out test set.

Fitting the three SVMs takes several minutes, which is too long to do in front
of an audience, so the first run fits them and saves them to
results/best_model/svm_*.joblib. Every run after that loads them and finishes
in a couple of seconds.

    python improvements/demo_best.py            # headline numbers + samples
    python improvements/demo_best.py --images my_photos/   # your own images
    python improvements/demo_best.py --errors   # show what it gets wrong
    python improvements/demo_best.py --show     # open the figure
    python improvements/demo_best.py --refit    # force a refit
"""

import argparse
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from joblib import dump, load as jload
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
FEAT = ROOT / "results" / "best_model"
FIGS = ROOT / "results" / "figures"

BACKBONES = ["efficientnetv2", "convnext_tiny", "swin_tiny"]
# timm id and native input resolution, matching best_model.py
BACKBONE_CFG = {
    "efficientnetv2": ("efficientnetv2_rw_s.ra2_in1k", 288),
    "convnext_tiny":  ("convnext_tiny.fb_in1k", 224),
    "swin_tiny":      ("swin_tiny_patch4_window7_224.ms_in1k", 224),
}
PARAMS = {"C": 10.0, "gamma": "scale"}       # chosen by cross-validation
CLASSES = ["undamaged", "damaged"]
SEED = 229


def feats(name, split):
    p = FEAT / f"{name}_tta_{split}.npy"
    if not p.exists():
        raise SystemExit(
            f"{p} missing.\nBuild the feature cache first:\n"
            f"  python improvements/best_model.py")
    return np.load(p)


def get_svms(refit=False):
    """Load the three fitted SVMs, fitting and caching them on first use."""
    y_pool = np.load(DATA / "y_pool.npy")
    tr = np.flatnonzero(~np.load(DATA / "pool_is_val.npy"))
    models = {}
    for name in BACKBONES:
        path = FEAT / f"svm_{name}.joblib"
        if path.exists() and not refit:
            models[name] = jload(path)
            continue
        print(f"  fitting SVM for {name} on {len(tr)} images "
              f"(one-off, ~2-3 min)...", flush=True)
        t0 = time.time()
        clf = SVC(kernel="rbf", probability=True, random_state=SEED,
                  **PARAMS).fit(feats(name, "pool")[tr], y_pool[tr])
        dump(clf, path)
        models[name] = clf
        print(f"    done in {time.time()-t0:.0f}s -> cached at {path.name}")
    return models


def classify_folder(path, models, show=False):
    """Run the full ensemble on arbitrary image files, live."""
    import timm, torch
    import torch.nn.functional as F
    from PIL import Image

    path = Path(path)
    files = ([path] if path.is_file()
             else sorted(f for f in path.iterdir()
                         if f.suffix.lower() in
                         (".jpg", ".jpeg", ".png", ".bmp", ".webp")))
    if not files:
        raise SystemExit(f"no images found in {path}")
    print(f"\n  classifying {len(files)} image(s) from {path}\n")

    imgs = np.stack([np.asarray(Image.open(f).convert("RGB")
                                .resize((224, 224), Image.BILINEAR),
                                dtype=np.uint8) for f in files])

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    probs = []
    for name in BACKBONES:
        timm_id, size = BACKBONE_CFG[name]
        net = timm.create_model(timm_id, pretrained=True, num_classes=0)
        net.eval().to(dev)
        cfg = timm.data.resolve_data_config({}, model=net)
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(dev)
        std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(dev)
        with torch.no_grad():
            xb = torch.from_numpy(imgs).to(dev).permute(0, 3, 1, 2).float() / 255
            if size != 224:
                xb = F.interpolate(xb, size=(size, size), mode="bilinear",
                                   align_corners=False)
            xb = (xb - mean) / std
            f = (net(xb) + net(torch.flip(xb, dims=[3]))) / 2   # TTA
        probs.append(models[name].predict_proba(f.cpu().numpy()))
        del net
        torch.cuda.empty_cache()

    P = sum(probs) / len(probs)
    print(f"  {'file':<34}{'verdict':>12}{'confidence':>13}")
    print("  " + "-" * 59)
    for f, p in zip(files, P):
        print(f"  {f.name[:34]:<34}{CLASSES[p.argmax()].upper():>12}"
              f"{p.max():>12.1%}")

    cols = min(4, len(files))
    nrows = int(np.ceil(len(files) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 3.2, nrows * 3.6),
                             squeeze=False)
    for ax in np.ravel(axes):
        ax.axis("off")
    for ax, im, p, f in zip(np.ravel(axes), imgs, P, files):
        ax.imshow(im)
        ax.set_title(f"{f.name[:22]}\n{CLASSES[p.argmax()]}  p={p.max():.2f}",
                     fontsize=9,
                     color="#b3512f" if p.argmax() == 1 else "#2f7d32")
    fig.suptitle("Final model on your own images", fontsize=13)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "08_final_custom.png"
    fig.savefig(out, dpi=95)
    plt.close(fig)
    print(f"\n  figure: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--errors", action="store_true",
                    help="show the most confident mistakes instead of a random sample")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--show", action="store_true")
    ap.add_argument("--refit", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--images", help="classify your own image file or folder")
    args = ap.parse_args()

    print("=" * 70)
    print("  FINAL MODEL - ensemble of three frozen backbones + RBF SVMs")
    print("  EfficientNetV2 (2021) | ConvNeXt (2022) | Swin Transformer (2021)")
    print("  test-time augmentation on, probabilities averaged")
    print("=" * 70)

    models = get_svms(args.refit)

    if args.images:
        classify_folder(args.images, models, args.show)
        return

    y = np.load(DATA / "y_test.npy")

    probs, singles = [], {}
    for name in BACKBONES:
        p = models[name].predict_proba(feats(name, "test"))
        probs.append(p)
        singles[name] = accuracy_score(y, p.argmax(1))
    P = sum(probs) / len(probs)
    pred = P.argmax(1)

    acc = accuracy_score(y, pred)
    cm = confusion_matrix(y, pred)
    rep = classification_report(y, pred, target_names=CLASSES,
                                output_dict=True, zero_division=0)
    auc = roc_auc_score(y, P[:, 1])
    base = max((y == 0).mean(), (y == 1).mean())

    print(f"\n  Held-out test set: {len(y)} images never seen during training,")
    print("  and that the original paper never evaluated on at all.\n")
    for name in BACKBONES:
        print(f"    {name:<16} alone   {singles[name]:.1%}")
    print(f"    {'ENSEMBLE':<16}         {acc:.1%}      <- the final model")
    print(f"\n  majority-class baseline  {base:.1%}")
    print(f"  the paper's backbone      83.8%")
    print(f"  ROC AUC                   {auc:.4f}")
    print(f"\n  confusion matrix")
    print(f"                   pred undamaged   pred damaged")
    print(f"    true undamaged   {cm[0,0]:>12}   {cm[0,1]:>12}")
    print(f"    true damaged     {cm[1,0]:>12}   {cm[1,1]:>12}")
    print(f"\n  damaged precision {rep['damaged']['precision']:.1%}   "
          f"recall {rep['damaged']['recall']:.1%}")

    errs = [p.argmax(1) != y for p in probs]
    shared = int(np.logical_and.reduce(errs).sum())
    print(f"\n  each model alone is wrong on "
          f"{', '.join(str(int(e.sum())) for e in errs)} images")
    print(f"  all three are wrong on the same {shared}")
    print("  -> they fail on different photographs, which is why voting helps")

    # ---- picture ----
    X = np.load(DATA / "X_test.npy", mmap_mode="r")
    if args.errors:
        rows = np.flatnonzero(pred != y)
        rows = rows[np.argsort(-P[rows].max(1))][:args.n]
        title = "Where the final model fails - most confident mistakes"
    else:
        rng = np.random.default_rng(args.seed)
        rows = rng.choice(len(y), args.n, replace=False)
        title = f"Final model on unseen images - {acc:.1%} over all {len(y)}"

    cols = 4
    nrows = int(np.ceil(len(rows) / cols))
    fig, axes = plt.subplots(nrows, cols, figsize=(cols * 3.2, nrows * 3.6))
    for ax in np.ravel(np.atleast_2d(axes)):
        ax.axis("off")
    print(f"\n  {'image':>8}  {'true':>10}  {'predicted':>10}  {'conf':>6}  ok")
    print("  " + "-" * 48)
    for ax, i in zip(np.ravel(np.atleast_2d(axes)), rows):
        ok = pred[i] == y[i]
        ax.imshow(np.asarray(X[i]))
        ax.set_title(f"true: {CLASSES[y[i]]}\n"
                     f"pred: {CLASSES[pred[i]]}  p={P[i].max():.2f}",
                     fontsize=9, color="#2f7d32" if ok else "#b3512f")
        print(f"  {i:>8}  {CLASSES[y[i]]:>10}  {CLASSES[pred[i]]:>10}  "
              f"{P[i].max():>6.1%}  {'YES' if ok else 'NO'}")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / ("08_final_errors.png" if args.errors else "08_final_demo.png")
    fig.savefig(out, dpi=95)
    plt.close(fig)
    print(f"\n  figure: {out}")
    if args.show:
        import subprocess, sys, os
        try:
            if sys.platform == "win32":
                os.startfile(out)
            else:
                subprocess.Popen(
                    ["open" if sys.platform == "darwin" else "xdg-open", str(out)],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass


if __name__ == "__main__":
    main()
