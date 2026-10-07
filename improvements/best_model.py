"""
The best model we can build, with no obligation to the paper's choices.

`repro/` is constrained: it matches the paper's backbone, hyperparameters,
dataset size and input resolution so the comparison is like-for-like. Nothing
here is constrained. The only rules are the ones that keep the result honest:
the 1,460-image test split is touched exactly once, at the end, and every
choice is made without looking at it.

What this does differently from everything before it
----------------------------------------------------
1. ALL available training data - 11,219 images, not the paper's 5,321.

2. Modern backbones - EfficientNetV2, ConvNeXt, Swin. The paper's InceptionV3
   is 2015; these are 2021-22 and were worth +4.4 points in modern_backbones.py.

3. Test-time augmentation - each image is passed through the backbone twice,
   once normally and once horizontally mirrored, and the two feature vectors
   are averaged. A damaged wall is still damaged when mirrored, so this is a
   free reduction in variance.

4. MODEL SELECTION BY CROSS-VALIDATION, which is the point of this script.
   modern_backbones.py found ResNet50 and EfficientNetV2 tied exactly on the
   592-image validation set (0.867) yet differed by 3.7 points on test - the
   validation split is too small to choose between models, and picking by it
   is close to a coin flip. Here the backbone AND the SVM hyperparameters are
   chosen by k-fold cross-validation over the training set, which uses
   thousands of held-out images instead of 592.

5. An ensemble of every backbone, reported alongside, so we can say whether
   combining them beats the cross-validated single model.

Run:  python improvements/best_model.py
      python improvements/best_model.py --skip-extract
      python improvements/best_model.py --folds 5 --no-tta
"""

import argparse
import json
import time
import zipfile
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
EXTRACT = RAW / "extracted"
DATA = ROOT / "data" / "processed"
FEAT = ROOT / "results" / "best_model"
RESULTS = ROOT / "results"
SRC = EXTRACT / "task2_X_train.npy"

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False
SEED = 229

BACKBONES = {
    "efficientnetv2": ("efficientnetv2_rw_s.ra2_in1k", 288),
    "convnext_tiny":  ("convnext_tiny.fb_in1k", 224),
    "swin_tiny":      ("swin_tiny_patch4_window7_224.ms_in1k", 224),
}

# SVM grid searched by cross-validation, not by peeking at validation or test.
GRID = [{"C": 1.0, "gamma": "scale"},
        {"C": 10.0, "gamma": "scale"},
        {"C": 1.0, "gamma": 1e-4},
        {"C": 10.0, "gamma": 1e-4}]


def ensure_source():
    if SRC.exists():
        return
    zp = RAW / "phi_task2_2.zip"
    if not zp.exists():
        raise SystemExit(f"{zp} not found - see the README for the download.")
    print(f"  extracting {SRC.name} (7.1 GB, a few minutes)...", flush=True)
    with zipfile.ZipFile(zp) as zf:
        name = next(n for n in zf.namelist() if n.endswith("X_train.npy"))
        with zf.open(name) as s, open(SRC, "wb") as d:
            while chunk := s.read(1 << 23):
                d.write(chunk)


MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


@torch.no_grad()
def features_for(model, mean, std, size, imgs_u8, tta, batch=24):
    """Backbone features for a uint8 RGB array, optionally averaged with the
    horizontally mirrored version (test-time augmentation)."""
    out = None
    for i in range(0, len(imgs_u8), batch):
        xb = torch.from_numpy(
            np.ascontiguousarray(imgs_u8[i:i + batch])).to(DEV)
        xb = xb.permute(0, 3, 1, 2).float().div_(255.0)
        if size != xb.shape[-1]:
            xb = F.interpolate(xb, size=(size, size), mode="bilinear",
                               align_corners=False)
        xb = (xb - mean) / std
        f = model(xb)
        if tta:
            f = (f + model(torch.flip(xb, dims=[3]))) / 2
        f = f.cpu().numpy()
        if out is None:
            out = np.empty((len(imgs_u8), f.shape[1]), dtype=np.float32)
        out[i:i + len(f)] = f
    return out


@torch.no_grad()
def extract_all(name, tta):
    """Pool (11,811) + val + test features for one backbone."""
    timm_id, size = BACKBONES[name]
    tag = f"{name}{'_tta' if tta else ''}"
    paths = {k: FEAT / f"{tag}_{k}.npy" for k in ("pool", "val", "test")}
    if all(p.exists() for p in paths.values()):
        print(f"    cached")
        return
    FEAT.mkdir(parents=True, exist_ok=True)

    model = timm.create_model(timm_id, pretrained=True, num_classes=0)
    model.eval().to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(DEV)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(DEV)

    # the two curated splits are already uint8 RGB on disk
    for split in ("val", "test"):
        if paths[split].exists():
            continue
        X = np.load(DATA / f"X_{split}.npy", mmap_mode="r")
        t0 = time.time()
        np.save(paths[split], features_for(model, mean, std, size,
                                           np.asarray(X), tta))
        print(f"    {split}: {len(X)} imgs in {time.time()-t0:.0f}s")

    # the pool needs de-normalising from the raw distribution
    if not paths["pool"].exists():
        ensure_source()
        x = np.load(SRC, mmap_mode="r")
        n = len(x)
        out = None
        t0 = time.time()
        for i in range(0, n, 240):
            j = min(i + 240, n)
            chunk = np.asarray(x[i:j], dtype=np.float32) + MEAN_BGR
            chunk = np.ascontiguousarray(chunk[..., ::-1])
            chunk = np.clip(np.rint(chunk), 0, 255).astype(np.uint8)
            f = features_for(model, mean, std, size, chunk, tta)
            if out is None:
                out = np.empty((n, f.shape[1]), dtype=np.float32)
            out[i:j] = f
            if i % 2400 == 0:
                print(f"      pool {i}/{n}", flush=True)
        np.save(paths["pool"], out)
        print(f"    pool: {n} imgs in {time.time()-t0:.0f}s")
    del model
    torch.cuda.empty_cache()


def load(name, split, tta):
    tag = f"{name}{'_tta' if tta else ''}"
    return np.load(FEAT / f"{tag}_{split}.npy")


def cv_score(X, y, params, folds, seed=SEED, cap=5000):
    """Mean accuracy over k stratified folds. Capped at `cap` samples so the
    search stays affordable - SVM training is roughly quadratic in n."""
    if len(y) > cap:
        rng = np.random.default_rng(seed)
        keep = []
        for c in (0, 1):
            pool = np.flatnonzero(y == c)
            keep.append(rng.choice(pool, round(cap * len(pool) / len(y)),
                                   replace=False))
        idx = np.concatenate(keep)
        X, y = X[idx], y[idx]
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(X, y):
        m = SVC(kernel="rbf", random_state=seed, **params).fit(X[tr], y[tr])
        accs.append(accuracy_score(y[te], m.predict(X[te])))
    return float(np.mean(accs)), float(np.std(accs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(BACKBONES))
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--skip-extract", action="store_true")
    ap.add_argument("--no-tta", action="store_true")
    args = ap.parse_args()
    tta = not args.no_tta

    y_pool = np.load(DATA / "y_pool.npy")
    is_val = np.load(DATA / "pool_is_val.npy")
    tr_rows = np.flatnonzero(~is_val)
    y_val = np.load(DATA / "y_val.npy")
    y_test = np.load(DATA / "y_test.npy")

    print(f"device={DEV}   TTA={'on' if tta else 'off'}   "
          f"{args.folds}-fold CV")
    print(f"training images: {len(tr_rows)}   val: {len(y_val)}   "
          f"test: {len(y_test)} (touched once, at the end)\n")

    if not args.skip_extract:
        for name in args.models:
            print(f"[extract] {name}")
            extract_all(name, tta)
        print()

    # ---- model selection: cross-validation on the TRAINING set only ----
    print("Model selection by cross-validation (test set not involved)")
    best = (-1, None, None)
    cv_table = {}
    for name in args.models:
        Ftr = load(name, "pool", tta)[tr_rows]
        ytr = y_pool[tr_rows]
        for params in GRID:
            t0 = time.time()
            m, s = cv_score(Ftr, ytr, params, args.folds)
            key = f"{name} C={params['C']} gamma={params['gamma']}"
            cv_table[key] = {"cv_mean": m, "cv_std": s}
            print(f"  {key:<46} {m:.4f} +-{s:.4f}   ({time.time()-t0:.0f}s)")
            if m > best[0]:
                best = (m, name, params)
        del Ftr

    cv_acc, name, params = best
    print(f"\n  WINNER: {name}, C={params['C']}, gamma={params['gamma']}  "
          f"(CV {cv_acc:.4f})")

    # ---- fit the winner on all training data, evaluate once ----
    Fp = load(name, "pool", tta)
    Ftr, ytr = Fp[tr_rows], y_pool[tr_rows]
    print(f"\n  fitting on all {len(ytr)} images...", flush=True)
    t0 = time.time()
    clf = SVC(kernel="rbf", probability=True, random_state=SEED,
              **params).fit(Ftr, ytr)
    print(f"  done in {time.time()-t0:.0f}s")

    pv = clf.predict_proba(load(name, "val", tta))
    pt = clf.predict_proba(load(name, "test", tta))
    va = accuracy_score(y_val, pv.argmax(1))
    ta = accuracy_score(y_test, pt.argmax(1))
    cm = confusion_matrix(y_test, pt.argmax(1))
    rep = classification_report(y_test, pt.argmax(1),
                                target_names=["undamaged", "damaged"],
                                output_dict=True, zero_division=0)
    auc = roc_auc_score(y_test, pt[:, 1])

    print(f"\n{'='*62}\n  BEST MODEL: {name} + RBF SVM, TTA={'on' if tta else 'off'}")
    print(f"{'='*62}")
    print(f"  validation      {va:.1%}")
    print(f"  HELD-OUT TEST   {ta:.1%}      ROC AUC {auc:.4f}")
    print(f"  confusion       {cm.tolist()}")
    print(f"  damaged: precision {rep['damaged']['precision']:.1%}  "
          f"recall {rep['damaged']['recall']:.1%}")
    print(f"  baseline        {max((y_test==0).mean(),(y_test==1).mean()):.1%}")

    payload = {
        "backbone": name, "svm_params": {k: str(v) for k, v in params.items()},
        "tta": tta, "n_train": int(len(ytr)), "folds": args.folds,
        "cv_accuracy": cv_acc, "cv_table": cv_table,
        "val_acc": float(va), "test_acc": float(ta), "roc_auc": float(auc),
        "test_confusion": cm.tolist(),
        "damaged_precision": round(rep["damaged"]["precision"], 4),
        "damaged_recall": round(rep["damaged"]["recall"], 4),
    }

    # ---- ensemble of every backbone, for comparison ----
    if len(args.models) > 1:
        probs, singles = [], {}
        for n2 in args.models:
            Fp2 = load(n2, "pool", tta)
            c2 = SVC(kernel="rbf", probability=True, random_state=SEED,
                     **params).fit(Fp2[tr_rows], y_pool[tr_rows])
            p2 = c2.predict_proba(load(n2, "test", tta))
            probs.append(p2)
            singles[n2] = float(accuracy_score(y_test, p2.argmax(1)))
            del Fp2
        P = sum(probs) / len(probs)
        ens = accuracy_score(y_test, P.argmax(1))
        errs = [p.argmax(1) != y_test for p in probs]
        shared = int(np.logical_and.reduce(errs).sum())
        print(f"\n  Ensemble of {len(args.models)} backbones: test {ens:.1%}")
        for k, v in singles.items():
            print(f"    {k:<16} alone {v:.1%}")
        print(f"    all wrong on the same {shared} images")
        payload["ensemble"] = {"test_acc": float(ens), "singles": singles,
                               "all_wrong_on": shared}

    out = RESULTS / "best_model.json"
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
