# Structural Damage Image Classification — Reproduction Study

Binary classification of structural photographs as **damaged** or **undamaged**,
reproducing the results of:

> M. Ho and J. Troncoso, *Structural Damage Image Classification*,
> CS229 Final Project, Stanford University, December 2018.

Machine Learning (UE24CS352A) mini-project.

---

## What this repository contains

A from-scratch PyTorch reimplementation of the paper's six models, plus its
bias/variance diagnostic and error analysis. The original authors' 2018
TensorFlow 1.x code does not run on current software and was used only as a
specification; none of it is included here. Every script under `repro/` was
written for this study.

The reproduction also goes beyond the paper in four ways:

1. **A real held-out test set.** The paper kept none ("we decided not to set
   aside images for testing"), so every number it reports is a validation
   number that also guided its modelling choices. We evaluate on the official
   PHI-Net 1,460-image test split.
2. **Input resolution.** The paper fed 224×224 images to a network pretrained
   at 299×299. Resizing recovers **+2.2 points** of validation accuracy.
3. **A larger SVM.** The authors capped their SVM at 3,000 images; we show
   this was a memory limit and that the full training set changes the result.
4. **A different bias/variance conclusion**, supported by a learning curve
   with error bars — see Findings below.

---

## Results

Validation accuracy on 592 images. Majority-class baseline: **53.4%**.

| Model | Paper train | Paper val | Ours train | Ours val |
|---|---|---|---|---|
| K-nearest neighbours (k=5) | 67.4% | 53.7% | 71.3% | 50.8% |
| Logistic regression | 99.4% | 54.7% | 99.8% | 54.0% |
| SVM (RBF, C=1, γ=0.001) | 99.7% | 50.7% | 95.1% | 55.9% |
| **InceptionV3, frozen + FC/softmax** | **81.0%** | **83.0%** | **88.6%** | **79.6%** |
| *— same, 299×299 input (ours)* | — | — | 87.6% | **81.8%** |
| *— SVM on all 5,321 images (ours)* | — | — | 94.9% | 57.8% |

**Held-out test set (1,460 images, not used by the paper): 81.2%**

Confusion matrices:

```
Paper §5.3 (1000 images):  [[448,  86],   accuracy 79.7%
                            [117, 349]]

Ours, validation (592):    [[255,  61],   accuracy 79.6%
                            [ 60, 216]]

Ours, held-out test (1460):[[581, 134],   accuracy 81.2%
                            [140, 605]]
```

A note on method: the saved splits are sorted by class, so taking the first N
rows is not a sample. All subsampling in this repository is stratified with a
fixed seed, and `baselines.py` prints the class balance it actually trained on.

The paper's Table 1 reports 83.0%, but its own §5.3 confusion matrix implies
**79.7%** — which our reproduction matches to within 0.1 points.

---

## Findings

**1. The paper's central claim reproduces.** Classical models on raw pixels are
indistinguishable from a constant guess (50–54% against a 53.4% baseline)
despite 97–99% training accuracy. Transfer learning reaches ~80%.

**2. Classical models memorise without learning.** The SVM reaches 95.1%
training accuracy and 55.9% validation — 2.5 points above a 53.4% baseline.
With 150,528 features and 5,321 examples there are 28× more parameters than
data points, so memorisation is free and generalisation is not.

**3. We disagree with the paper's bias diagnosis.** The paper read its learning
curve as "clearly a bias issue" and added capacity in response. Our curve shows
a **26-point** train/validation gap at 200 images, falling to 8.6 points at
5,321 — the signature of variance, not bias. At full size the validation curve
has flattened at ~80%, so the limit is the frozen ImageNet features and label
noise, not model capacity.

**4. The errors are structural, not random.** False positives are dominated by
high-texture undamaged surfaces (brickwork, tiling, tree branches) and by
genuinely **mislabelled** images. False negatives are dominated by **wide
shots** where damage is signalled by context — shoring props, scaffolding,
rubble — rather than by crack texture. The model learned close-up crack
appearance. The paper listed this as unfinished work; `results/figures/`
demonstrates it.

**5. The dataset contains label noise.** Example: source row 7818 of
`task2_X_train.npy` is labelled *undamaged* but shows a bridge girder with
spalled concrete and exposed reinforcement. This places a ceiling on achievable
accuracy and is consistent with the paper's own remark that "some images were
incorrectly labeled".

---

## Dataset

PEER Hub ImageNet (Φ-Net), **Task 2 — Damage State**, CC BY-NC-SA 4.0.
<https://apps.peer.berkeley.edu/phi-net/>

Download both Task 2 archives into `data/raw/`:

```bash
mkdir -p data/raw
curl -L -o data/raw/phi_task2_1.zip "https://apps.peer.berkeley.edu/phichallenge/dataset/kfs55dd33"
curl -L -o data/raw/phi_task2_2.zip "https://apps.peer.berkeley.edu/phichallenge/dataset/task2_damage_state_2.zip"
```

Expected sizes: 175,290,492 and 1,410,780,428 bytes. **Verify these** — a
truncated or duplicated download produces a zip that opens in some tools and
silently corrupts results.

Two properties of the distribution are undocumented and must be handled:

* Images are stored **BGR with the ImageNet per-channel mean already
  subtracted** (range −123.68 to +151.06). Recover true RGB with
  `round(x + [103.939, 116.779, 123.68])[..., ::-1]`. Clipping to `[0,255]`
  without this destroys about half of every image.
* Labels are `0 = Damaged, 1 = Undamaged` — the **opposite** of the paper's
  convention. We use the paper's (`y = 1 - y_raw`). Verified by inspection;
  see `repro/verify_labels.py`.

The public release (13,271 images) is larger than the 2018 challenge set. We
subsample to the paper's exact 5,913 images and class counts, seed 229.

---

## Setup

Requires Python 3.12 and an NVIDIA GPU (tested on an RTX 4050, 6 GB).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 .venv
uv pip install -r requirements.txt
```

---

## Running

```bash
# 1. Build the splits (expects both zips in data/raw/)
.venv/bin/python repro/prepare_data.py

# 2. Verify the data before trusting anything trained on it
.venv/bin/python repro/verify_labels.py      # label polarity
.venv/bin/python repro/raw_label_check.py    # raw labels, verbatim
.venv/bin/python repro/baselines.py --verify # GPU maths vs scikit-learn

# 3. Classical baselines (report §4.1–4.3)
.venv/bin/python repro/baselines.py --model all

# 4. Transfer learning (report §4.4)
.venv/bin/python repro/inception_transfer.py --extract --train
.venv/bin/python repro/inception_transfer.py --extract --train --size 299

# 5. Diagnostics (report §5.2–5.3)
.venv/bin/python repro/learning_curve.py --seeds 3
.venv/bin/python repro/error_analysis.py

# optional: export every image as a browsable JPEG
.venv/bin/python repro/export_images.py
```

### Live demo

`repro/demo.py` loads the saved classifier head and runs in a couple of
seconds — no training, no network.

```bash
.venv/bin/python repro/demo.py                 # classify 8 random unseen images
.venv/bin/python repro/demo.py --score         # full metrics on the test set
.venv/bin/python repro/demo.py --errors        # the model's worst mistakes
.venv/bin/python repro/demo.py --image photo.jpg   # any image file
```

Add `--show` to open the figure, `--seed N` to reproduce a particular sample,
`--size 224` to use the paper's input resolution instead of 299.

Runtimes on the reference machine: data preparation ~3 min, baselines ~4 min,
feature extraction 11 s, head training ~20 s, learning curve ~6 min.

---

## Learning path

`docs/COURSE.md` rebuilds this project from scratch as a nine-module course,
with verifiable checkpoints, deliberate traps and a hint ladder. Work in
`practice/` (git-ignored) and keep notes in `practice/NOTES.md`.

---

## Layout

```
repro/           all analysis code (written for this study)
  prepare_data.py       splits, de-normalisation, label flip
  verify_labels.py      label-polarity verification
  raw_label_check.py    raw labels verbatim + source-index mapping
  baselines.py          KNN / logistic regression / SVM
  inception_transfer.py frozen InceptionV3 + FC head
  learning_curve.py     bias/variance diagnostic
  error_analysis.py     confusion matrices and error galleries
  export_images.py      dataset → browsable JPEGs
docs/paper_spec.md   full specification extracted from the paper
results/             metrics (JSON), cached features, figures
data/                dataset (git-ignored)
```

---

## Implementation notes

**Cached bottleneck features.** The backbone is frozen, so its output per image
never changes. Extracting the 2048-d features once turns 4,000 training
iterations into matrix multiplies on a 5321×2048 array — seconds instead of
hours. This is what TensorFlow for Poets, the tutorial the authors followed,
calls "bottlenecks".

**GPU distance computations.** The flattened feature matrix is 5321 × 150,528
(3.2 GB float32); scikit-learn's brute-force KNN drove a 15 GB machine into
swap, and libsvm requires float64 (6.4 GB) — almost certainly why the authors
capped their SVM at 3,000 images. Since KNN and the RBF SVM need only pairwise
distances, we compute those on the GPU in blocks and pass scikit-learn a
precomputed kernel. `--verify` checks this reproduces scikit-learn exactly.

**Numerical care.** `‖a−b‖² = ‖a‖² + ‖b‖² − 2a·b` loses precision when ‖a‖² ≈
45,000 and ‖a−b‖² ≈ 8,000. Centring the data (which leaves distances unchanged)
reduces the error 20×. TF32 is disabled so float32 matmuls keep a full mantissa.

**Preprocessing deviation.** The paper used TF-slim weights expecting `x/128−1`.
We use torchvision weights, which were trained with ImageNet mean/std
normalisation, and normalise accordingly — preprocessing must match the weights
being loaded.

---

## Not reproduced

* MobileNetV1 (paper: 80.0% / 67.0%)
* SVM on layer-288 InceptionV3 activations (paper: 95.0% / 75.0%)
* Fine-tuning experiments from the poster (100% / 74.2%, and with augmentation
  77.2% / 72.8%)

---

## License

Analysis code: MIT. The Φ-Net dataset is CC BY-NC-SA 4.0 and is not
redistributed here.
