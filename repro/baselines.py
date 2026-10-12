"""
Step 3: Classical ML baselines on raw pixels (report sections 4.1-4.3).

Reproduces rows 1-3 of Table 1:

    Model                        Train    Val
    K-nearest neighbours (k=5)   67.4%    53.7%
    Logistic regression          99.4%    54.7%
    SVM (RBF)                    99.7%    50.7%

Each image is flattened to 224*224*3 = 150,528 features scaled to [0,1],
matching the original `knn_logistic.py` (`images / images.max()`).

WHY THIS USES THE GPU
---------------------
The feature matrix is 5321 x 150528 float32 = 3.2 GB. Running scikit-learn's
brute-force KNN on it drove this 15 GB machine to 7.2 GB RSS and heavy swap
thrashing (scikit-learn upcasts float32 blocks to float64 inside
`euclidean_distances` for precision). libsvm is worse: it requires float64
outright, which is 6.4 GB for the same data -- which is almost certainly why
the original authors capped their SVM at 3000 images.

Both KNN and the RBF SVM only ever need PAIRWISE DISTANCES, not the raw
features. So we compute the distance/Gram matrices on the GPU in blocks and
hand scikit-learn a precomputed kernel. Consequences:

  * KNN is exact brute force, same k, same Euclidean metric.
  * SVC(kernel="precomputed") is mathematically identical to
    SVC(kernel="rbf", gamma=g) but never materialises the 150k-dim features
    in float64, so we can run it on ALL 5321 images instead of 3000.
  * Host RAM stays under ~1 GB; peak VRAM ~3.6 GB of the 6.1 GB available.

`--verify` checks the GPU path against scikit-learn on a small subset.

Usage:
    python repro/baselines.py --model knn
    python repro/baselines.py --model logreg
    python repro/baselines.py --model svm
    python repro/baselines.py --model all
    python repro/baselines.py --verify
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"

# Paper's hyperparameters, report sections 4.1-4.3
KNN_K = 5
LOGREG_C = 1.0
SVM_C = 1.0
SVM_GAMMA = 0.001
SVM_N_SAMPLES_PAPER = 3000  # the authors' svm_classification.py cap

DEV = "cuda" if torch.cuda.is_available() else "cpu"

# Ampere+ GPUs silently use TF32 (10-bit mantissa) for float32 matmuls. Our
# dot products are over 150k dimensions, so we want the full 24-bit mantissa.
torch.backends.cuda.matmul.allow_tf32 = False


def mem_available_gb():
    """Free RAM in GB, or nan where the platform does not expose it.

    /proc/meminfo is Linux-only; these diagnostics are informational, so on
    other platforms we report nan rather than failing.
    """
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return float("nan")


def peak_rss_gb():
    """Peak resident memory in GB, or nan where unavailable.

    `resource` is Unix-only. Windows exposes the same figure through the Win32
    API, so fall back to that before giving up.
    """
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024 / 1024
    except ImportError:
        pass
    try:                                    # Windows
        import ctypes
        from ctypes import wintypes

        class _Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD),
                        ("PageFaultCount", wintypes.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]

        c = _Counters()
        c.cb = ctypes.sizeof(c)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return c.PeakWorkingSetSize / 2 ** 30
    except Exception:
        return float("nan")


def subsample_indices(y, limit, seed=229):
    """Pick `limit` rows at random, preserving the class ratio.

    The saved splits are SORTED by class (prepare_data.py sorts by source
    index, and the distribution groups the classes). So `X[:limit]` is not a
    sample -- it is almost entirely one class. The paper sliced a SHUFFLED
    array, so a stratified random draw is what reproduces its protocol.
    Indices are returned sorted, which keeps the memory-mapped reads sequential.
    """
    if limit is None or limit >= len(y):
        return None
    rng = np.random.default_rng(seed)
    keep = []
    for cls in np.unique(y):
        pool = np.flatnonzero(y == cls)
        k = int(round(limit * len(pool) / len(y)))
        keep.append(rng.choice(pool, min(k, len(pool)), replace=False))
    return np.sort(np.concatenate(keep))


def load_gpu(split, limit=None, block=256):
    """Stream a split onto the GPU as flattened float32 in [0,1].

    Host RAM never holds more than one `block` of images.
    """
    x = np.load(DATA / f"X_{split}.npy", mmap_mode="r")
    y = np.load(DATA / f"y_{split}.npy").astype(np.int64)
    idx = subsample_indices(y, limit)
    if idx is None:
        idx = np.arange(len(y))
    out = torch.empty((len(idx), 224 * 224 * 3), dtype=torch.float32,
                      device=DEV)
    for i in range(0, len(idx), block):
        sl = idx[i : i + block]
        chunk = torch.from_numpy(np.ascontiguousarray(x[sl])).to(DEV)
        out[i : i + len(sl)] = chunk.reshape(len(sl), -1).float().div_(255.0)
    return out, y[idx]


def load_host(split, limit=None, block=256):
    """Same thing but as a host float32 array (needed by sklearn's lbfgs)."""
    x = np.load(DATA / f"X_{split}.npy", mmap_mode="r")
    y = np.load(DATA / f"y_{split}.npy").astype(np.int64)
    idx = subsample_indices(y, limit)
    if idx is None:
        idx = np.arange(len(y))
    out = np.empty((len(idx), 224 * 224 * 3), dtype=np.float32)
    for i in range(0, len(idx), block):
        sl = idx[i : i + block]
        out[i : i + len(sl)] = np.asarray(x[sl], dtype=np.float32).reshape(
            len(sl), -1)
    out /= 255.0
    return out, y[idx]


def column_mean(X, block=512):
    """Column mean without a full-size temporary."""
    acc = torch.zeros(X.shape[1], dtype=torch.float64, device=X.device)
    for i in range(0, X.shape[0], block):
        j = min(i + block, X.shape[0])
        acc += X[i:j].sum(0, dtype=torch.float64)
    return (acc / X.shape[0]).float()


def center_(*tensors, mu):
    """Subtract a shared mean in place.

    ||a-b||^2 is invariant to translation, but the float32 error of the
    expansion ||a||^2 + ||b||^2 - 2a.b grows with the MAGNITUDE of those
    terms. Pixels live in [0,1], so ||a||^2 ~ 45,000 while ||a-b||^2 ~ 8,000:
    we lose most of our significant digits to the cancellation. Centring
    drops ||a||^2 to ~10,000 and cuts the error by about 4x, at no cost to
    correctness.
    """
    for t in tensors:
        t -= mu
    return tensors


def sq_norms(X, block=512):
    """Row-wise ||x||^2 without allocating a full-size temporary.

    `(X * X).sum(1)` looks harmless but materialises a second copy of X --
    3.2 GB here, which is what blew up the 6 GB GPU. Doing it per block keeps
    the temporary at `block x 150528`.
    """
    out = torch.empty(X.shape[0], dtype=X.dtype, device=X.device)
    for i in range(0, X.shape[0], block):
        j = min(i + block, X.shape[0])
        out[i:j] = X[i:j].pow(2).sum(1)
    return out


def sq_dists(A, B, block=1024):
    """Pairwise squared Euclidean distances, computed on the GPU in blocks.

    ||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b
    Returned on the GPU as (len(A), len(B)) float32.
    """
    an = sq_norms(A)
    bn = sq_norms(B)
    out = torch.empty((A.shape[0], B.shape[0]), dtype=torch.float32, device=DEV)
    for i in range(0, A.shape[0], block):
        j = min(i + block, A.shape[0])
        d = an[i:j, None] + bn[None, :] - 2.0 * (A[i:j] @ B.T)
        out[i:j] = d.clamp_(min=0)
    return out


def knn_predict(Xq, Xtr, ytr, k=KNN_K, block=512):
    """Exact brute-force KNN majority vote (binary labels)."""
    ytr_t = torch.from_numpy(ytr).to(DEV)
    preds = torch.empty(Xq.shape[0], dtype=torch.int64, device=DEV)
    an = sq_norms(Xq)
    bn = sq_norms(Xtr)
    for i in range(0, Xq.shape[0], block):
        j = min(i + block, Xq.shape[0])
        d = an[i:j, None] + bn[None, :] - 2.0 * (Xq[i:j] @ Xtr.T)
        nn = d.topk(k, dim=1, largest=False).indices       # (b, k)
        votes = ytr_t[nn].sum(1)                            # count of class 1
        preds[i:j] = (votes * 2 > k).long()                 # majority, k odd
    return preds.cpu().numpy()


def rbf_gram(A, B, gamma):
    """RBF kernel matrix, float64 on the host (what libsvm wants)."""
    return torch.exp(-gamma * sq_dists(A, B)).double().cpu().numpy()


def summarise(name, ytr, tr_pred, yva, va_pred, extra=None):
    row = {
        "model": name,
        "train_acc": round(float(metrics.accuracy_score(ytr, tr_pred)), 4),
        "val_acc": round(float(metrics.accuracy_score(yva, va_pred)), 4),
        "n_train": int(len(ytr)),
        "n_val": int(len(yva)),
        "val_confusion": metrics.confusion_matrix(yva, va_pred).tolist(),
        "majority_baseline_val": round(
            float(max((yva == 0).mean(), (yva == 1).mean())), 4
        ),
        "peak_rss_gb": round(peak_rss_gb(), 2),
    }
    if extra:
        row.update(extra)
    print(f"\n=== {name} ===")
    print(f"  train acc {row['train_acc']:.3f}   val acc {row['val_acc']:.3f}"
          f"   (majority baseline {row['majority_baseline_val']:.3f})")
    print(f"  val confusion [[TN FP],[FN TP]] = {row['val_confusion']}")
    print(f"  peak host RSS {row['peak_rss_gb']} GB"
          + (f"   peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GB"
             if DEV == "cuda" else ""))
    return row


def run_knn(results):
    t0 = time.time()
    Xtr, ytr = load_gpu("train")
    Xva, yva = load_gpu("val")
    center_(Xtr, Xva, mu=column_mean(Xtr))
    tr_pred = knn_predict(Xtr, Xtr, ytr)
    va_pred = knn_predict(Xva, Xtr, ytr)
    row = summarise(f"KNN (k={KNN_K})", ytr, tr_pred, yva, va_pred,
                    {"seconds": round(time.time() - t0, 1),
                     "backend": f"exact brute force on {DEV}"})
    results.append(row)
    del Xtr, Xva
    torch.cuda.empty_cache()


def run_logreg(results):
    t0 = time.time()
    Xtr, ytr = load_host("train")
    # L2 is sklearn's default; passing penalty="l2" is deprecated in 1.8+.
    # C=1.0 and max_iter=1000 match the authors' knn_logistic.py.
    clf = LogisticRegression(C=LOGREG_C, solver="lbfgs", max_iter=1000)
    clf.fit(Xtr, ytr)
    tr_pred = clf.predict(Xtr)
    del Xtr
    Xva, yva = load_host("val")
    va_pred = clf.predict(Xva)
    del Xva
    row = summarise("Logistic Regression", ytr, tr_pred, yva, va_pred,
                    {"seconds": round(time.time() - t0, 1),
                     "n_iter": int(clf.n_iter_[0]),
                     "coef_dtype": str(clf.coef_.dtype),
                     "converged": bool(clf.n_iter_[0] < 1000),
                     # lbfgs can abort its line search under memory pressure
                     # and return after a handful of iterations, leaving the
                     # model badly under-trained. Flag it rather than let a
                     # silently wrong number into the results.
                     "suspect_early_stop": bool(clf.n_iter_[0] < 50)})
    if clf.n_iter_[0] < 50:
        print(f"  WARNING: lbfgs stopped after only {clf.n_iter_[0]} "
              f"iterations - this result is under-trained and should be "
              f"re-run on a less loaded machine.")
    results.append(row)


def run_svm(results, n_limit=None, tag=""):
    t0 = time.time()
    Xtr, ytr = load_gpu("train", limit=n_limit)
    Xva, yva = load_gpu("val")
    print(f"  [balance] trained on {len(ytr)} images: "
          f"{(ytr==1).sum()} damaged ({(ytr==1).mean():.1%}), "
          f"{(ytr==0).sum()} undamaged ({(ytr==0).mean():.1%})")
    center_(Xtr, Xva, mu=column_mean(Xtr))
    Ktr = rbf_gram(Xtr, Xtr, SVM_GAMMA)
    Kva = rbf_gram(Xva, Xtr, SVM_GAMMA)

    off_diag = Ktr[~np.eye(len(Ktr), dtype=bool)]
    print(f"  [kernel] K(x,x)=1.0 by definition; off-diagonal "
          f"mean={off_diag.mean():.3e} max={off_diag.max():.3e}")

    clf = SVC(kernel="precomputed", C=SVM_C)
    clf.fit(Ktr, ytr)
    tr_pred = clf.predict(Ktr)
    va_pred = clf.predict(Kva)
    row = summarise(f"SVM (RBF, C={SVM_C}, gamma={SVM_GAMMA}){tag}",
                    ytr, tr_pred, yva, va_pred,
                    {"seconds": round(time.time() - t0, 1),
                     "n_support_vectors": int(clf.n_support_.sum()),
                     "kernel_offdiag_mean": float(off_diag.mean()),
                     "kernel_offdiag_max": float(off_diag.max()),
                     "backend": "precomputed RBF Gram on GPU + libsvm"})
    results.append(row)
    del Xtr, Xva, Ktr, Kva
    torch.cuda.empty_cache()


def verify():
    """Check the GPU path reproduces scikit-learn exactly on a small subset."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics.pairwise import rbf_kernel

    n = 400
    Xh, yh = load_host("train", limit=n)
    Xq, yq = load_host("val", limit=120)
    Xg = torch.from_numpy(Xh).to(DEV)
    Xqg = torch.from_numpy(Xq).to(DEV)
    center_(Xg, Xqg, mu=column_mean(Xg))

    sk = KNeighborsClassifier(n_neighbors=KNN_K, algorithm="brute").fit(Xh, yh)
    sk_pred = sk.predict(Xq)
    gpu_pred = knn_predict(Xqg, Xg, yh)
    agree = (sk_pred == gpu_pred).mean()
    print(f"KNN  : GPU vs sklearn agreement on {len(yq)} queries = {agree:.4f}")

    # sklearn computes this in float64 from the uncentred features, so it is
    # the reference. Per-entry relative error is what matters for the SVM,
    # since a kernel entry only ever enters the decision function scaled by
    # its own magnitude.
    K_sk = rbf_kernel(Xq, Xh, gamma=SVM_GAMMA)
    K_gpu = rbf_gram(Xqg, Xg, SVM_GAMMA)
    abs_err = np.abs(K_sk - K_gpu).max()
    rel_entry = np.abs(K_sk - K_gpu).max() / max(K_sk.max(), 1e-30)
    per_entry = (np.abs(K_sk - K_gpu) / np.maximum(K_sk, 1e-30)).max()
    print(f"RBF  : max abs err {abs_err:.3e} | "
          f"rel to K_max {rel_entry:.3e} | max per-entry rel {per_entry:.3e}")

    assert agree == 1.0, "GPU KNN disagrees with sklearn"
    assert per_entry < 1e-2, "GPU RBF kernel differs from sklearn"
    print("verification passed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="all",
                    choices=["knn", "logreg", "svm", "all"])
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()

    print(f"device={DEV}  RAM available={mem_available_gb():.1f} GB")
    if args.verify:
        verify()
        return

    RESULTS.mkdir(exist_ok=True)
    out_path = RESULTS / "baselines.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else []

    def drop(substr):
        return [r for r in results if substr not in r["model"].lower()]

    if args.model in ("knn", "all"):
        results = drop("knn")
        run_knn(results)
        out_path.write_text(json.dumps(results, indent=2))
    if args.model in ("logreg", "all"):
        results = drop("logistic")
        run_logreg(results)
        out_path.write_text(json.dumps(results, indent=2))
    if args.model in ("svm", "all"):
        results = drop("svm")
        # Faithful to the authors' code (3000-image cap) ...
        run_svm(results, n_limit=int(0.9 * SVM_N_SAMPLES_PAPER),
                tag=" [paper protocol, 2700 train]")
        # ... and on the full training set, which they could not afford.
        run_svm(results, n_limit=None, tag=" [full 5321 train]")
        out_path.write_text(json.dumps(results, indent=2))

    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
