"""
Step 2: Turn the raw PHI-Net Task 2 zips into the splits the paper used.

Reproduces the dataset described in Ho & Troncoso (CS229, 2018):
    5913 labelled images (224x224 8-bit RGB), 2727 damaged (46%) /
    3186 undamaged (54%), split 90/10:
        train 5321 = 2870 undamaged + 2451 damaged
        val    592 =  316 undamaged +  276 damaged
    The authors kept no held-out test set ("because of the limited number of
    samples"), so their "validation accuracy" is what Table 1 reports. We DO
    keep the official PHI-Net test split (1460 images) as a bonus held-out set,
    which is a strict improvement on their protocol.

The public Phi-Net Task 2 release is LARGER than the 2018 challenge set
(11811 train + 1460 test = 13271 images). We therefore subsample 5913 images
at the paper's exact class ratio so the numbers are comparable.

LABEL CONVENTION
----------------
Raw file (per PHI-Net README):   0 = Damaged,   1 = Undamaged
Paper's convention (inferred
from its confusion matrix):      0 = Undamaged, 1 = Damaged
We FLIP to the paper's convention everywhere downstream.

Run:  python repro/prepare_data.py
"""

import json
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
EXTRACT = RAW / "extracted"
OUT = ROOT / "data" / "processed"

# Paper's numbers, Section 1 of the report.  The authors did a plain 90/10
# slice of their (shuffled) array rather than a stratified split, which is why
# the validation set is 46.6% damaged while the pool is 46.1%.  We hard-code
# their exact counts so our splits are the same size as theirs.
N_TOTAL = 5913
N_DAMAGED = 2727        # label 1
N_UNDAMAGED = 3186      # label 0
N_TRAIN_DAMAGED = 2451
N_TRAIN_UNDAMAGED = 2870
N_VAL_DAMAGED = 276
N_VAL_UNDAMAGED = 316
SEED = 229  # CS229 :)


def unzip_all():
    EXTRACT.mkdir(parents=True, exist_ok=True)
    for z in sorted(RAW.glob("*.zip")):
        with zipfile.ZipFile(z) as zf:
            for name in zf.namelist():
                if not name.endswith(".npy"):
                    continue
                target = EXTRACT / Path(name).name
                if target.exists():
                    print(f"  skip (exists): {target.name}")
                    continue
                print(f"  extracting {name} from {z.name} ...")
                with zf.open(name) as src, open(target, "wb") as dst:
                    while chunk := src.read(1 << 22):
                        dst.write(chunk)


def to_paper_labels(y_onehot):
    """(N,2) one-hot with raw convention -> (N,) int with paper convention."""
    raw = np.argmax(y_onehot, axis=1).astype(np.int64)  # 0=Damaged, 1=Undamaged
    return 1 - raw  # 0=Undamaged, 1=Damaged


# The raw .npy images are NOT plain pixels. They are stored in BGR channel
# order with the ImageNet per-channel mean already subtracted (Caffe style),
# so values run from -123.68 to +151.06. Adding the means back recovers exact
# integers in [0,255]; reversing the channel axis recovers RGB. Verified
# round-trip-exact on the test split.
MEAN_BGR = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def images_to_uint8(x_mmap, indices, batch=256):
    """Pull `indices` out of the memory-mapped array as true RGB uint8.

    The raw train split is 11811 x 224 x 224 x 3 float32 = 1.8 GB, which does
    not fit comfortably in this machine's free RAM. We memory-map it and copy
    out only the rows we need, in batches, undoing the mean-subtraction and
    BGR ordering as we go. uint8 is lossless here and 4x smaller.
    """
    out = np.empty((len(indices), 224, 224, 3), dtype=np.uint8)
    for i in range(0, len(indices), batch):
        sl = indices[i : i + batch]
        chunk = np.asarray(x_mmap[sl], dtype=np.float32) + MEAN_BGR  # de-mean
        chunk = chunk[..., ::-1]                                     # BGR -> RGB
        out[i : i + len(sl)] = np.clip(np.rint(chunk), 0, 255).astype(np.uint8)
    return out


def stratified_take(labels, n_per_class, rng):
    """Pick exactly n_per_class[c] indices of each class c."""
    picked = []
    for cls, n in n_per_class.items():
        pool = np.flatnonzero(labels == cls)
        if len(pool) < n:
            raise SystemExit(
                f"Only {len(pool)} images of class {cls} available, need {n}."
            )
        picked.append(rng.choice(pool, size=n, replace=False))
    idx = np.concatenate(picked)
    rng.shuffle(idx)
    return idx


def main():
    print("[1/4] Extracting .npy files from zips")
    unzip_all()

    xtr_p = EXTRACT / "task2_X_train.npy"
    ytr_p = EXTRACT / "task2_y_train.npy"
    xte_p = EXTRACT / "task2_X_test.npy"
    yte_p = EXTRACT / "task2_y_test.npy"

    missing = [p.name for p in (xtr_p, ytr_p, xte_p, yte_p) if not p.exists()]
    if missing:
        raise SystemExit(
            f"Missing {missing} in {EXTRACT}.\n"
            "Did you download BOTH Task 2 zips into data/raw/ ?\n"
            "  Task 2_1 (test):  https://apps.peer.berkeley.edu/phichallenge/dataset/kfs55dd33\n"
            "  Task 2_2 (train): https://apps.peer.berkeley.edu/phichallenge/dataset/yyaa44rrr"
        )

    print("[2/4] Reading labels")
    y_train_raw = np.load(ytr_p)
    y_test_raw = np.load(yte_p)
    y_pool = to_paper_labels(y_train_raw)
    y_test = to_paper_labels(y_test_raw)

    n_un, n_dam = int((y_pool == 0).sum()), int((y_pool == 1).sum())
    print(f"      full train pool: {len(y_pool)} images "
          f"({n_dam} damaged {n_dam/len(y_pool):.1%}, "
          f"{n_un} undamaged {n_un/len(y_pool):.1%})")
    print(f"      official test set: {len(y_test)} images "
          f"({int((y_test==1).sum())} damaged, {int((y_test==0).sum())} undamaged)")

    print(f"[3/4] Subsampling to the paper's {N_TOTAL} images "
          f"({N_DAMAGED} damaged / {N_UNDAMAGED} undamaged)")
    rng = np.random.default_rng(SEED)
    sel = stratified_take(y_pool, {0: N_UNDAMAGED, 1: N_DAMAGED}, rng)
    y_sel = y_pool[sel]

    # 90/10 split using the report's exact per-class counts
    val_local = stratified_take(
        y_sel, {0: N_VAL_UNDAMAGED, 1: N_VAL_DAMAGED}, rng
    )
    val_mask = np.zeros(len(sel), dtype=bool)
    val_mask[val_local] = True

    tr_idx_sorted = np.sort(sel[~val_mask])
    va_idx_sorted = np.sort(sel[val_mask])

    print(f"      train {len(tr_idx_sorted)} "
          f"({int((y_pool[tr_idx_sorted]==0).sum())} undamaged / "
          f"{int((y_pool[tr_idx_sorted]==1).sum())} damaged)  "
          f"| val {len(va_idx_sorted)} "
          f"({int((y_pool[va_idx_sorted]==0).sum())} undamaged / "
          f"{int((y_pool[va_idx_sorted]==1).sum())} damaged)")
    print(f"      paper:  train 5321 (2870 / 2451)  | val 592 (316 / 276)")

    print("[4/4] Materialising image arrays as uint8 (this takes a minute)")
    OUT.mkdir(parents=True, exist_ok=True)
    x_pool = np.load(xtr_p, mmap_mode="r")
    probe = np.asarray(x_pool[:64], dtype=np.float32) + MEAN_BGR
    print(f"      source dtype={x_pool.dtype} shape={x_pool.shape}")
    print(f"      after un-normalising: range=[{probe.min():.2f}, {probe.max():.2f}] "
          f"integral={np.allclose(probe, np.rint(probe), atol=1e-2)}")
    if not (-0.5 <= probe.min() and probe.max() <= 255.5):
        raise SystemExit("Train split is normalised differently than the test "
                         "split - inspect before continuing.")

    np.save(OUT / "X_train.npy", images_to_uint8(x_pool, tr_idx_sorted))
    np.save(OUT / "y_train.npy", y_pool[tr_idx_sorted])
    np.save(OUT / "X_val.npy", images_to_uint8(x_pool, va_idx_sorted))
    np.save(OUT / "y_val.npy", y_pool[va_idx_sorted])
    del x_pool

    x_te = np.load(xte_p, mmap_mode="r")
    np.save(OUT / "X_test.npy", images_to_uint8(x_te, np.arange(len(y_test))))
    np.save(OUT / "y_test.npy", y_test)
    del x_te

    meta = {
        "seed": SEED,
        "label_convention": {"0": "undamaged", "1": "damaged"},
        "n_train": int(len(tr_idx_sorted)),
        "n_val": int(len(va_idx_sorted)),
        "n_test": int(len(y_test)),
        "train_damaged_frac": float((y_pool[tr_idx_sorted] == 1).mean()),
        "val_damaged_frac": float((y_pool[va_idx_sorted] == 1).mean()),
        "note": "Subsampled from the modern PHI-Net Task 2 release to match the "
                "2018 challenge size (5913) and class ratio (46/54) used by "
                "Ho & Troncoso, CS229 2018.",
    }
    (OUT / "split_meta.json").write_text(json.dumps(meta, indent=2))
    print("\nDone. Wrote to", OUT)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
