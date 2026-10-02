"""
Train the production model: best configuration, all available data.

The reproduction (`inception_transfer.py`) deliberately restricts itself to the
paper's 5,913 images so the comparison is like-for-like. This script drops that
restriction and trains the best model we can:

    * every training image the modern release offers (11,219 after the 592
      validation images are excluded)
    * 299x299 input, InceptionV3's native pretraining resolution
    * the same frozen backbone + linear head, same optimiser as the paper

Evaluated on two sets the model never sees:
    * the same 592-image validation set used throughout, so every number in
      this project remains comparable
    * the official PHI-Net 1,460-image test split, which is the honest score

Run:  python repro/train_full.py
      python repro/train_full.py --seeds 5 --size 224
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics

from inception_transfer import BATCH, DEV, ITERS, LR, SEED

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
FEAT = ROOT / "results" / "features"
RESULTS = ROOT / "results"


def load_pool(size):
    tag = "" if size == 224 else f"_{size}"
    f = FEAT / f"incv3_pool{tag}.npy"
    if not f.exists():
        raise SystemExit(
            f"{f} missing - run:\n"
            f"  python repro/extract_pool_features.py --size {size}")
    feats = np.load(f)
    y = np.load(DATA / "y_pool.npy")
    is_val = np.load(DATA / "pool_is_val.npy")
    return feats, y, is_val


def load_split(split, size):
    tag = "" if size == 224 else f"_{size}"
    return (np.load(FEAT / f"incv3_{split}{tag}.npy"),
            np.load(DATA / f"y_{split}.npy"))


def train_head(Xtr, Ytr, seed, iters=ITERS, lr=LR, batch=BATCH):
    torch.manual_seed(seed)
    head = nn.Linear(2048, 2).to(DEV)
    opt = torch.optim.SGD(head.parameters(), lr=lr)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(iters):
        sel = torch.randint(0, len(Ytr), (batch,), generator=g, device=DEV)
        loss = F.cross_entropy(head(Xtr[sel]), Ytr[sel])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head


@torch.no_grad()
def score(head, feats, y):
    pred = head(torch.from_numpy(feats).to(DEV)).argmax(1).cpu().numpy()
    return metrics.accuracy_score(y, pred), pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=299, choices=[224, 299])
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    feats, y_pool, is_val = load_pool(args.size)
    train_rows = np.flatnonzero(~is_val)
    f_val, y_val = load_split("val", args.size)
    f_test, y_test = load_split("test", args.size)

    print(f"device={DEV}  input={args.size}x{args.size}")
    print(f"  training images : {len(train_rows)} "
          f"({(y_pool[train_rows]==1).sum()} damaged, "
          f"{(y_pool[train_rows]==0).sum()} undamaged)")
    print(f"  validation      : {len(y_val)} (held out of training)")
    print(f"  held-out test   : {len(y_test)} (official PHI-Net test split)")
    print(f"  optimiser       : SGD lr={LR} batch={BATCH} iters={ITERS}\n")

    Xtr = torch.from_numpy(feats[train_rows]).to(DEV)
    Ytr = torch.from_numpy(y_pool[train_rows]).to(DEV)

    val_accs, test_accs, best = [], [], (-1, None, None)
    for k in range(args.seeds):
        head = train_head(Xtr, Ytr, SEED + k)
        va, _ = score(head, f_val, y_val)
        te, te_pred = score(head, f_test, y_test)
        val_accs.append(va)
        test_accs.append(te)
        print(f"  seed {SEED+k}:  val {va:.3f}   test {te:.3f}")
        if va > best[0]:
            best = (va, head, te_pred)

    _, head, te_pred = best
    cm = metrics.confusion_matrix(y_test, te_pred)
    rep = metrics.classification_report(
        y_test, te_pred, target_names=["undamaged", "damaged"],
        output_dict=True, zero_division=0)

    print(f"\n=== Production model (all {len(train_rows)} images, "
          f"{args.size}px) ===")
    print(f"  validation : {np.mean(val_accs):.3f} +- {np.std(val_accs):.3f}")
    print(f"  TEST       : {np.mean(test_accs):.3f} +- {np.std(test_accs):.3f}")
    print(f"  test confusion [[TN FP],[FN TP]] = {cm.tolist()}")
    print(f"  damaged precision {rep['damaged']['precision']:.3f}  "
          f"recall {rep['damaged']['recall']:.3f}")
    print(f"  majority baseline on test: "
          f"{max((y_test==0).mean(), (y_test==1).mean()):.3f}")

    payload = {
        "model": f"InceptionV3 frozen + FC/softmax, all data, {args.size}px",
        "n_train": int(len(train_rows)),
        "input_size": args.size,
        "seeds": args.seeds,
        "val_acc_mean": round(float(np.mean(val_accs)), 4),
        "val_acc_std": round(float(np.std(val_accs)), 4),
        "test_acc_mean": round(float(np.mean(test_accs)), 4),
        "test_acc_std": round(float(np.std(test_accs)), 4),
        "test_confusion": cm.tolist(),
        "damaged_precision": round(rep["damaged"]["precision"], 4),
        "damaged_recall": round(rep["damaged"]["recall"], 4),
        "lr": LR, "batch": BATCH, "iters": ITERS,
    }
    (RESULTS / "production_model.json").write_text(json.dumps(payload, indent=2))
    torch.save(head.state_dict(), RESULTS / f"head_full_{args.size}.pt")
    print(f"\n  wrote {RESULTS/'production_model.json'}")
    print(f"  wrote {RESULTS/f'head_full_{args.size}.pt'}")


if __name__ == "__main__":
    main()
