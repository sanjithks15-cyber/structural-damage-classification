"""
BEYOND THE PAPER (1): does a modern backbone beat InceptionV3?

The 2018 paper used InceptionV3, published in 2015. It is the only CNN they
tried apart from MobileNetV1. Everything since -- EfficientNet (2019),
Vision Transformers (2020), Swin (2021), ConvNeXt (2022) -- postdates their
work, so this comparison was not available to them.

The question is narrow and fair: holding the data, the split, the classifier
and the hyperparameters fixed, how much of our error is the fault of the
feature extractor?

Method
------
For each backbone:
  1. run it frozen over train/val/test at its own native input resolution
  2. train the identical FC+softmax head (SGD, lr 0.01, batch 100, 4000 iters)
  3. also fit an RBF SVM on the same features, since we found that beats the
     linear head on InceptionV3 features
  4. report validation and held-out test accuracy

Then ensemble the best backbones by averaging their probabilities. Ensembles
only help when members make different mistakes, so the script also reports
how much their errors overlap.

Everything here is additive: no existing file or result is modified. Outputs
land in results/backbones/ and results/modern_backbones.json.

Run:  python repro/modern_backbones.py
      python repro/modern_backbones.py --models convnext_tiny efficientnetv2_rw_s
      python repro/modern_backbones.py --skip-extract      # reuse cached features
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
FEAT = ROOT / "results" / "backbones"
RESULTS = ROOT / "results"

DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = False

# The paper's optimiser settings, unchanged, so the only variable is the
# feature extractor.
LR, BATCH, ITERS, SEED = 0.01, 100, 4000, 229

# name -> (timm id, input resolution, year, one-line description)
BACKBONES = {
    "inceptionv3":  ("inception_v3.tv_in1k",                    299, 2015,
                     "the paper's backbone (control)"),
    "resnet50":     ("resnet50.a1_in1k",                        224, 2015,
                     "contemporary of InceptionV3, modern training recipe"),
    "efficientnetv2": ("efficientnetv2_rw_s.ra2_in1k",          288, 2021,
                     "compound-scaled CNN"),
    "swin_tiny":    ("swin_tiny_patch4_window7_224.ms_in1k",    224, 2021,
                     "hierarchical vision transformer"),
    "convnext_tiny": ("convnext_tiny.fb_in1k",                  224, 2022,
                     "CNN redesigned with transformer-era choices"),
}


@torch.no_grad()
def extract(name, batch=32):
    """Run one frozen backbone over all three splits and cache the features."""
    timm_id, size, _, _ = BACKBONES[name]
    model = timm.create_model(timm_id, pretrained=True, num_classes=0)
    model.eval().to(DEV)
    for p in model.parameters():
        p.requires_grad_(False)

    cfg = timm.data.resolve_data_config({}, model=model)
    mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(DEV)
    std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(DEV)

    FEAT.mkdir(parents=True, exist_ok=True)
    dims = None
    for split in ("train", "val", "test"):
        out_path = FEAT / f"{name}_{split}.npy"
        if out_path.exists():
            print(f"    {split}: cached")
            dims = np.load(out_path, mmap_mode="r").shape[1]
            continue
        X = np.load(DATA / f"X_{split}.npy", mmap_mode="r")
        n = len(X)
        feats = None
        t0 = time.time()
        for i in range(0, n, batch):
            j = min(i + batch, n)
            xb = torch.from_numpy(np.ascontiguousarray(X[i:j])).to(DEV)
            xb = xb.permute(0, 3, 1, 2).float().div_(255.0)
            if size != 224:
                xb = F.interpolate(xb, size=(size, size), mode="bilinear",
                                   align_corners=False)
            out = model((xb - mean) / std)
            if feats is None:
                dims = out.shape[1]
                feats = np.empty((n, dims), dtype=np.float32)
            feats[i:j] = out.cpu().numpy()
        np.save(out_path, feats)
        print(f"    {split}: {feats.shape} in {time.time()-t0:.0f}s")
    del model
    torch.cuda.empty_cache()
    return dims


def load(name, split):
    return (np.load(FEAT / f"{name}_{split}.npy"),
            np.load(DATA / f"y_{split}.npy"))


def train_linear(Ftr, ytr, dim, seed=SEED):
    torch.manual_seed(seed)
    Xtr = torch.from_numpy(Ftr).to(DEV)
    Ytr = torch.from_numpy(ytr).to(DEV)
    head = nn.Linear(dim, 2).to(DEV)
    opt = torch.optim.SGD(head.parameters(), lr=LR)
    g = torch.Generator(device=DEV).manual_seed(seed)
    for _ in range(ITERS):
        sel = torch.randint(0, len(Ytr), (BATCH,), generator=g, device=DEV)
        loss = F.cross_entropy(head(Xtr[sel]), Ytr[sel])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return head


@torch.no_grad()
def linear_probs(head, F_):
    return torch.softmax(head(torch.from_numpy(F_).to(DEV)), 1).cpu().numpy()


def evaluate(name, seeds=3):
    dim = np.load(FEAT / f"{name}_train.npy", mmap_mode="r").shape[1]
    Ftr, ytr = load(name, "train")
    Fva, yva = load(name, "val")
    Fte, yte = load(name, "test")

    # linear head, averaged over seeds, keeping the best-on-validation probs
    v, t, best = [], [], (-1, None, None)
    for k in range(seeds):
        head = train_linear(Ftr, ytr, dim, SEED + k)
        pv, pt = linear_probs(head, Fva), linear_probs(head, Fte)
        av = accuracy_score(yva, pv.argmax(1))
        v.append(av)
        t.append(accuracy_score(yte, pt.argmax(1)))
        if av > best[0]:
            best = (av, pv, pt)
    lin = {"val_mean": float(np.mean(v)), "val_std": float(np.std(v)),
           "test_mean": float(np.mean(t)), "test_std": float(np.std(t))}

    # RBF SVM on the same features
    svm = SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
              random_state=SEED).fit(Ftr, ytr)
    pv_s, pt_s = svm.predict_proba(Fva), svm.predict_proba(Fte)
    sv = {"val": float(accuracy_score(yva, pv_s.argmax(1))),
          "test": float(accuracy_score(yte, pt_s.argmax(1)))}

    return {"dim": int(dim), "linear": lin, "svm": sv}, best[2], pt_s, yte


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(BACKBONES))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    print(f"device={DEV}   classifier settings fixed at the paper's "
          f"(SGD lr={LR}, batch={BATCH}, {ITERS} iters)\n")

    results, test_probs = {}, {}
    for name in args.models:
        timm_id, size, year, note = BACKBONES[name]
        print(f"[{name}]  {timm_id}  ({year}, {size}px) - {note}")
        if not args.skip_extract:
            extract(name)
        res, pt_lin, pt_svm, yte = evaluate(name, args.seeds)
        results[name] = {**res, "timm_id": timm_id, "year": year, "size": size}
        test_probs[name] = {"linear": pt_lin, "svm": pt_svm}
        print(f"    features {res['dim']}-d")
        print(f"    linear head  val {res['linear']['val_mean']:.3f}"
              f"+-{res['linear']['val_std']:.3f}   "
              f"test {res['linear']['test_mean']:.3f}"
              f"+-{res['linear']['test_std']:.3f}")
        print(f"    SVM          val {res['svm']['val']:.3f}   "
              f"test {res['svm']['test']:.3f}\n")

    # ---- summary table, ordered by validation accuracy of the better head ----
    print("=" * 74)
    print(f"  {'backbone':<16}{'year':>5}{'dim':>6}{'linear test':>14}"
          f"{'SVM test':>11}{'best':>9}")
    print("-" * 74)
    order = sorted(results, key=lambda n: -max(
        results[n]["linear"]["val_mean"], results[n]["svm"]["val"]))
    for n in order:
        r = results[n]
        best = max(r["linear"]["test_mean"], r["svm"]["test"])
        print(f"  {n:<16}{r['year']:>5}{r['dim']:>6}"
              f"{r['linear']['test_mean']:>13.1%}{r['svm']['test']:>11.1%}"
              f"{best:>9.1%}")
    print("=" * 74)

    # ---- ensemble the top three by validation ----
    if len(order) >= 2:
        top = order[:3]
        print(f"\n  Ensemble of {', '.join(top)} (average of SVM probabilities)")
        P = sum(test_probs[n]["svm"] for n in top) / len(top)
        acc = accuracy_score(yte, P.argmax(1))
        print(f"    test accuracy {acc:.1%}")
        errs = {n: test_probs[n]["svm"].argmax(1) != yte for n in top}
        allwrong = np.logical_and.reduce(list(errs.values())).sum()
        for n in top:
            print(f"    {n} alone wrong on {errs[n].sum()}")
        print(f"    all {len(top)} wrong on the same {allwrong} images "
              f"-> ensembles only help where this is small")
        cm = confusion_matrix(yte, P.argmax(1))
        rep = classification_report(yte, P.argmax(1),
                                    target_names=["undamaged", "damaged"],
                                    output_dict=True, zero_division=0)
        results["ensemble"] = {
            "members": top, "test_acc": float(acc),
            "test_confusion": cm.tolist(),
            "damaged_recall": round(rep["damaged"]["recall"], 4),
            "damaged_precision": round(rep["damaged"]["precision"], 4),
            "all_members_wrong_on": int(allwrong),
        }
        print(f"    confusion {cm.tolist()}   "
              f"damaged recall {rep['damaged']['recall']:.1%}")

    out = RESULTS / "modern_backbones.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
