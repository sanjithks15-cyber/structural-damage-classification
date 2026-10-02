"""
Step 4: Transfer learning with InceptionV3 (report section 4.4).

Target, Table 1 row 5:   train 81.0%   val 83.0%

METHOD (report S4.4 + poster)
-----------------------------
Take InceptionV3 pretrained on ImageNet, freeze every layer, drop the 1000-way
ImageNet classifier, and train a single fully-connected layer with softmax on
top of the 2048-d pooled features. Optimiser: plain gradient descent,
lr = 0.01, batch 100, 4000 iterations. This is the "TensorFlow for Poets"
recipe the authors used.

WHY WE CACHE FEATURES
---------------------
The backbone is frozen, so its output for a given image never changes. Running
it once and caching the 2048-d vectors turns 4000 training iterations from
4000 CNN forward passes into 4000 matrix multiplies on a 5321x2048 array --
seconds instead of hours. TensorFlow for Poets does exactly this and calls
them "bottlenecks". The cached features are reused by later steps (the
layer-288 SVM, the learning curve).

DEVIATION FROM THE PAPER
------------------------
The authors used TF-slim InceptionV3 weights, which expect inputs scaled to
[-1,1] (their `x/128 - 1`). We use torchvision's InceptionV3 weights, which
were trained with ImageNet mean/std normalisation. Preprocessing must match
the weights you load -- feeding [-1,1] to torchvision weights degrades the
features. We therefore use torchvision's normalisation and say so.

Usage:
    python repro/inception_transfer.py --extract          # cache features
    python repro/inception_transfer.py --train            # train the head
    python repro/inception_transfer.py --extract --train
    python repro/inception_transfer.py --extract --size 299   # variant
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
FEAT = ROOT / "results" / "features"
RESULTS = ROOT / "results"

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Paper's optimiser settings (poster: "GradientDescent with a learning rate of
# 0.01 ... Batch size of 100 with 4000 iterations").
LR = 0.01
BATCH = 100
ITERS = 4000
SEED = 229

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def backbone():
    """Frozen InceptionV3 trunk that outputs the 2048-d pooled feature."""
    m = inception_v3(weights=Inception_V3_Weights.IMAGENET1K_V1)
    m.fc = nn.Identity()      # drop the 1000-way ImageNet classifier
    m.aux_logits = False
    m.AuxLogits = None
    m.eval().to(DEV)
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def extract(split, size=224, batch=32):
    X = np.load(DATA / f"X_{split}.npy", mmap_mode="r")
    y = np.load(DATA / f"y_{split}.npy")
    net = backbone()
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)
    feats = np.empty((len(y), 2048), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(y), batch):
        j = min(i + batch, len(y))
        xb = torch.from_numpy(np.ascontiguousarray(X[i:j])).to(DEV)
        xb = xb.permute(0, 3, 1, 2).float().div_(255.0)
        if size != 224:
            xb = F.interpolate(xb, size=(size, size), mode="bilinear",
                               align_corners=False)
        xb = (xb - mean) / std
        feats[i:j] = net(xb).cpu().numpy()
        if i % (batch * 40) == 0:
            print(f"    {split} {i}/{len(y)}", flush=True)
    FEAT.mkdir(parents=True, exist_ok=True)
    tag = "" if size == 224 else f"_{size}"
    np.save(FEAT / f"incv3_{split}{tag}.npy", feats)
    print(f"  {split}: {feats.shape} in {time.time()-t0:.0f}s -> "
          f"incv3_{split}{tag}.npy")
    del net
    torch.cuda.empty_cache()
    return feats, y


def load_feats(split, size=224):
    tag = "" if size == 224 else f"_{size}"
    f = FEAT / f"incv3_{split}{tag}.npy"
    if not f.exists():
        raise SystemExit(f"{f} missing - run with --extract first")
    return np.load(f), np.load(DATA / f"y_{split}.npy")


def train_head(size=224, lr=LR, iters=ITERS, batch=BATCH, seed=SEED,
               n_train=None, quiet=False):
    """Train the single FC+softmax layer on cached bottleneck features."""
    torch.manual_seed(seed)
    Ftr, ytr = load_feats("train", size)
    Fva, yva = load_feats("val", size)
    if n_train is not None:
        # Stratified subsample: the paper "ensured that ratio of damaged vs.
        # undamaged stayed constant (46/54) regardless of training sample
        # size", which matters because an unstratified draw moves the
        # majority-class baseline and makes points on the learning curve
        # incomparable to each other.
        rng = np.random.default_rng(seed)
        keep = []
        for cls in (0, 1):
            pool = np.flatnonzero(ytr == cls)
            k = int(round(n_train * len(pool) / len(ytr)))
            keep.append(rng.choice(pool, min(k, len(pool)), replace=False))
        idx = np.sort(np.concatenate(keep))
        Ftr, ytr = Ftr[idx], ytr[idx]

    Xtr = torch.from_numpy(Ftr).to(DEV)
    Ytr = torch.from_numpy(ytr).to(DEV)
    Xva = torch.from_numpy(Fva).to(DEV)
    Yva = torch.from_numpy(yva).to(DEV)

    head = nn.Linear(2048, 2).to(DEV)
    opt = torch.optim.SGD(head.parameters(), lr=lr)   # plain GD, as in paper
    g = torch.Generator(device=DEV).manual_seed(seed)

    history = []
    for it in range(1, iters + 1):
        sel = torch.randint(0, len(Ytr), (batch,), generator=g, device=DEV)
        loss = F.cross_entropy(head(Xtr[sel]), Ytr[sel])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 200 == 0 or it == 1:
            with torch.no_grad():
                tr = (head(Xtr).argmax(1) == Ytr).float().mean().item()
                va = (head(Xva).argmax(1) == Yva).float().mean().item()
            history.append({"iter": it, "loss": loss.item(),
                            "train_acc": tr, "val_acc": va})
            if not quiet:
                print(f"    iter {it:>5}  loss {loss.item():.4f}  "
                      f"train {tr:.3f}  val {va:.3f}", flush=True)

    with torch.no_grad():
        tr_pred = head(Xtr).argmax(1).cpu().numpy()
        va_pred = head(Xva).argmax(1).cpu().numpy()
    return head, history, (ytr, tr_pred, yva, va_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--iters", type=int, default=ITERS)
    args = ap.parse_args()

    print(f"device={DEV}  input size={args.size}")

    if args.extract:
        print("Extracting frozen InceptionV3 bottleneck features:")
        for s in ("train", "val", "test"):
            extract(s, size=args.size)

    if args.train:
        from sklearn import metrics
        print(f"Training FC+softmax head: lr={LR} batch={BATCH} "
              f"iters={args.iters} (paper: GD, 0.01, 100, 4000)")
        head, history, (ytr, tr_pred, yva, va_pred) = train_head(
            size=args.size, iters=args.iters)

        tr_acc = metrics.accuracy_score(ytr, tr_pred)
        va_acc = metrics.accuracy_score(yva, va_pred)
        cm = metrics.confusion_matrix(yva, va_pred)
        print(f"\n=== InceptionV3 frozen + FC/softmax (input {args.size}) ===")
        print(f"  train acc {tr_acc:.3f}   val acc {va_acc:.3f}")
        print(f"  paper:    train 0.810    val 0.830")
        print(f"  val confusion [[TN FP],[FN TP]] = {cm.tolist()}")
        print(f"  majority baseline on val: "
              f"{max((yva==0).mean(),(yva==1).mean()):.3f}")

        RESULTS.mkdir(exist_ok=True)
        tag = "" if args.size == 224 else f"_{args.size}"
        out = RESULTS / f"inception_transfer{tag}.json"
        payload = {
            "model": f"InceptionV3 frozen + FC/softmax (input {args.size})",
            "train_acc": round(float(tr_acc), 4),
            "val_acc": round(float(va_acc), 4),
            "val_confusion": cm.tolist(),
            "paper_train_acc": 0.810,
            "paper_val_acc": 0.830,
            "lr": LR, "batch": BATCH, "iters": args.iters, "seed": SEED,
            "input_size": args.size,
            "history": history,
        }
        out.write_text(json.dumps(payload, indent=2))
        torch.save(head.state_dict(), RESULTS / f"head_incv3_{args.size}.pt")
        print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
