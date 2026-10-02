"""
Extract InceptionV3 features for the ENTIRE 11,811-image training pool.

Why: our reproduction uses 5,913 images to match the paper. The modern release
has more than twice that. This script makes the extra images usable so you can
answer, with a measurement rather than a guess: *does more data help?*

What it writes (to results/features/ and data/processed/):
    incv3_pool.npy     (11811, 2048) float32  - features for every pool image
    y_pool.npy         (11811,)      int64    - labels, paper convention
    pool_is_val.npy    (11811,)      bool     - True for the 592 images in our
                                                validation set

The mask matters. To compare fairly you must train on more images but evaluate
on the SAME validation set, otherwise you are changing two things at once.
Exclude every row where pool_is_val is True from training.

Needs ~7.1 GB of free disk temporarily for the source array; pass --keep to
leave it extracted afterwards.

Run:  python repro/extract_pool_features.py
"""

import argparse
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from inception_transfer import (DEV, IMAGENET_MEAN, IMAGENET_STD, backbone)
from prepare_data import MEAN_BGR

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
EXTRACT = RAW / "extracted"
DATA = ROOT / "data" / "processed"
FEAT = ROOT / "results" / "features"

SRC = EXTRACT / "task2_X_train.npy"


def ensure_source():
    if SRC.exists():
        print(f"  source already extracted: {SRC.name}")
        return
    zp = RAW / "phi_task2_2.zip"
    if not zp.exists():
        raise SystemExit(f"{zp} not found - see the README for the download.")
    print(f"  extracting {SRC.name} from {zp.name} (7.1 GB, a few minutes)...")
    with zipfile.ZipFile(zp) as zf:
        name = next(n for n in zf.namelist() if n.endswith("X_train.npy"))
        with zf.open(name) as s, open(SRC, "wb") as d:
            while chunk := s.read(1 << 23):
                d.write(chunk)
    print("  done")


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--keep", action="store_true",
                    help="keep the 7.1 GB source file afterwards")
    args = ap.parse_args()

    print("[1/3] Source array")
    ensure_source()

    print("[2/3] Labels and validation mask")
    y_raw = np.load(EXTRACT / "task2_y_train.npy")
    y_pool = (1 - np.argmax(y_raw, axis=1)).astype(np.int64)  # paper convention

    val_src = DATA / "val_src_index.npy"
    if not val_src.exists():
        raise SystemExit(
            "data/processed/val_src_index.npy missing - run "
            "repro/raw_label_check.py once to generate it.")
    is_val = np.zeros(len(y_pool), dtype=bool)
    is_val[np.load(val_src)] = True

    print(f"      pool {len(y_pool)} images: "
          f"{(y_pool==1).sum()} damaged ({(y_pool==1).mean():.1%}), "
          f"{(y_pool==0).sum()} undamaged ({(y_pool==0).mean():.1%})")
    print(f"      of these, {is_val.sum()} are our validation images "
          f"-> {(~is_val).sum()} usable for training")

    print(f"[3/3] Extracting features at {args.size}x{args.size}")
    x = np.load(SRC, mmap_mode="r")
    net = backbone()
    mean, std = IMAGENET_MEAN.to(DEV), IMAGENET_STD.to(DEV)
    feats = np.empty((len(y_pool), 2048), dtype=np.float32)
    t0 = time.time()
    for i in range(0, len(y_pool), args.batch):
        j = min(i + args.batch, len(y_pool))
        # same de-normalisation as prepare_data.py: undo the ImageNet mean
        # subtraction and the BGR ordering, back to true RGB in [0,255]
        chunk = np.asarray(x[i:j], dtype=np.float32) + MEAN_BGR
        chunk = np.ascontiguousarray(chunk[..., ::-1])
        xb = torch.from_numpy(chunk).to(DEV).permute(0, 3, 1, 2)
        xb = xb.clamp_(0, 255).div_(255.0)
        if args.size != 224:
            xb = F.interpolate(xb, size=(args.size, args.size),
                               mode="bilinear", align_corners=False)
        feats[i:j] = net((xb - mean) / std).cpu().numpy()
        if i % (args.batch * 60) == 0:
            print(f"      {i}/{len(y_pool)}", flush=True)
    print(f"      done in {time.time()-t0:.0f}s")

    FEAT.mkdir(parents=True, exist_ok=True)
    tag = "" if args.size == 224 else f"_{args.size}"
    np.save(FEAT / f"incv3_pool{tag}.npy", feats)
    np.save(DATA / "y_pool.npy", y_pool)
    np.save(DATA / "pool_is_val.npy", is_val)
    print(f"\n  wrote {FEAT / f'incv3_pool{tag}.npy'}  {feats.shape}")
    print(f"  wrote {DATA/'y_pool.npy'} and {DATA/'pool_is_val.npy'}")

    del net
    torch.cuda.empty_cache()
    if not args.keep:
        SRC.unlink()
        print(f"  removed {SRC.name} (7.1 GB reclaimed; pass --keep to retain)")


if __name__ == "__main__":
    main()
