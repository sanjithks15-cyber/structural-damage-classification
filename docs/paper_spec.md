# Reproduction target — specification

**Paper:** Minnie Ho (Intel) & Jorge Troncoso (Google), *Structural Damage Image
Classification*, CS229 Final Project, Stanford, December 2018.
Two artefacts: a 6-page report (`39.pdf`) and a 1-page poster (`39 (1).pdf`).
Original code: <https://github.com/jatron/structural-damage-recognition>
(this repository is a fork of it).

---

## 1. Dataset

| Property | Value |
|---|---|
| Source | PEER Hub ImageNet (Φ-Net) Challenge, **Task 2 — Damage State** |
| Size used in paper | 5,913 labelled images |
| Class 0 "undamaged" | 3,186 (54%) |
| Class 1 "damaged" | 2,727 (46%) |
| Image format | 224 × 224, 8-bit RGB |
| Held-out test set | **None** — "we decided not to set aside images for testing because of the limited number of samples" |

**Split (report §1), exact counts:**

| Split | Undamaged | Damaged | Total | % damaged |
|---|---|---|---|---|
| Train (90%) | 2,870 | 2,451 | 5,321 | 46.1% |
| Validation (10%) | 316 | 276 | 592 | 46.6% |

The counts are not a stratified split — they are a plain 90/10 slice of a
shuffled array (see `knn_logistic.py` in the original code:
`X_train = ltrain_images[:int(.9 * n_samples)]`).

### Deviations we are forced to make

1. **Dataset version.** The public Φ-Net Task 2 release is now 13,271 images
   (11,811 train + 1,460 test), not 5,913. The 2018 challenge subset is not
   recoverable. We subsample 5,913 at the paper's exact class counts, seed 229.
2. **Pool class ratio is mirrored.** The modern pool is 53.2% damaged /
   46.8% undamaged — the opposite of the paper's 46/54. The subsample fixes this.
3. **We keep a real held-out test set** (the official 1,460-image Φ-Net test
   split). The paper did not. This is a strict improvement on their protocol
   and is reported separately, never substituted for their "validation accuracy".

### Raw file format (discovered empirically, not documented)

The distributed `.npy` arrays are `float32`, **BGR** channel order, with the
ImageNet per-channel mean already subtracted. Range is `[-123.68, +151.06]`.

```
rgb_uint8 = round(x_float32 + [103.939, 116.779, 123.68])[..., ::-1]
```

Verified round-trip-exact. Clipping to `[0, 255]` without this correction
destroys roughly half of every image.

Label encoding in the raw file is `0 = Damaged, 1 = Undamaged` — the **opposite**
of the paper's convention, which the paper's confusion matrix row sums
(534 / 466 ≈ 54% / 46%) reveal to be `0 = Undamaged, 1 = Damaged`.
We use the paper's convention throughout: `y = 1 - y_raw`.

### Verifying the label flip (`repro/verify_labels.py`)

No accuracy metric can detect a label flip — swapping both class names leaves
every accuracy score unchanged and only corrupts the interpretation. So this
was checked directly, three ways:

| Check | Outcome |
|---|---|
| PHI-Net README (inside the zip) | `0: Damaged, 1: Undamaged` — **confirmed correct by inspection** |
| Official Task 2 page (lists UD first, then D) | weak/ambiguous; page ordering is not a label spec |
| Edge-energy proxy over 800 images | **wrong answer**: our damaged class has *lower* edge energy (Cohen's d = -0.25) |
| Visual inspection, our labels (24 images) | **decisive** — our class 1 is unambiguously damage |
| Visual inspection, raw labels (16 images) | 15/16 match the README; `y_raw=[1,0]` really is damaged |

There is no contradiction between the dataset and the paper: they simply use
opposite conventions, and `y = 1 - y_raw` reconciles them.

| | class 0 | class 1 |
|---|---|---|
| Raw file (README, correct) | Damaged | Undamaged |
| Paper, and our code | Undamaged | Damaged |

The edge-energy proxy failed because it measures surface *texture*, not damage:
damaged images are mostly close-ups of smooth walls with one thin crack, while
undamaged images include tile grids, brick courses, panelling and foliage. This
is an independent rediscovery of the paper's own error analysis (report S5.3,
"patterns can be mistaken as cracks, and vice versa"), whose false positives
were brick patterns, ridges and curtains.

Figures: `results/figures/01_label_polarity_check.png` (our convention),
`results/figures/02_raw_label_review.png` (raw labels, verbatim).

### Observed label noise

Source row **7818** of `task2_X_train.npy` is labelled `Undamaged` but shows a
bridge girder with spalled concrete and exposed reinforcement. Found in a
random sample of 16, i.e. roughly 6% in that sample. Concrete evidence for the
paper's remark (report S3) that "some images were incorrectly labeled", and a
reason no model on this task should be expected to reach 100%.

Traceability: `data/processed/train_src_index.npy` and `val_src_index.npy` map
each processed row back to its row in the original distribution, so any image
can be cited by its source index (written by `repro/raw_label_check.py`).

---

## 2. Preprocessing

| Model family | Preprocessing (report §3) |
|---|---|
| CNNs | `x = x/128 - 1`, giving range [-1, 1) |
| KNN / LogReg / SVM | scaled and flattened to a 150,528-dim vector (original code divides by `max()`, giving [0, 1]) |

---

## 3. Models and hyperparameters

| § | Model | Hyperparameters (report §4) |
|---|---|---|
| 4.1 | K-nearest neighbours | `k = 5` (sklearn default); no tuning performed |
| 4.2 | Logistic regression | L2 penalty, `C = 1.0`; `lbfgs`, `max_iter=1000` (from code) |
| 4.3 | SVM | RBF kernel, `C = 1.0`, `gamma = 0.001`; trained on 3,000 images in the original code, not all 5,913 |
| 4.4 | MobileNetV1.0 | pretrained on ImageNet, all layers frozen, new FC + softmax head |
| 4.4 | InceptionV3 (TF-for-Poets) | pretrained on ImageNet, frozen, new FC + softmax head. **GradientDescent, lr = 0.01, batch 100, 4,000 iterations** (poster) |
| 4.5 | SVM on InceptionV3 activations | features = output of **layer 288** of 311; RBF kernel with `gamma` tuned by model selection |
| 5.4 | InceptionV3 (Keras) | remove top layer, Flatten, Dense + softmax → **2,001,000 trainable parameters** → overfits |
| 5.4 | InceptionV3 (Keras + augmentation) | `ImageDataGenerator(width_shift=0.2, height_shift=0.2, horizontal_flip=True)`, fresh augmentation every iteration |

Optimiser note: the poster says gradient descent produced noisy accuracy
curves, so they later switched to Adam.

---

## 4. Results to reproduce

### Table 1 (report §5.1) — primary target

| Model | Train acc | Val acc |
|---|---|---|
| K-nearest neighbours (k=5) | 67.4% | 53.7% (test 52.9%) |
| Logistic regression | 99.4% | 54.7% |
| SVM, RBF kernel | 99.7% | 50.7% |
| MobileNetV1 | 80.0% | 67.0% |
| **InceptionV3** | **81.0%** | **83.0%** |
| SVM on layer-288 activations | 95.0% | 75.0% |

### Poster-only rows (not in the report's Table 1)

| Model | Train acc | Val acc |
|---|---|---|
| MobileNetV1.0 | 80.0% | 67.7% |
| SVM, RBF kernel | 99.8% | 50.7% |
| InceptionV3 (Keras) | 100.0% | 74.2% |
| InceptionV3 (Keras + augmentation) | 77.2% | 72.8% |

Note the report and poster disagree slightly (SVM 99.7 vs 99.8, MobileNet
67.0 vs 67.7). The report is the later, more authoritative document.

### Secondary artefacts

1. **Learning curve / bias-variance** (report Fig. 4): accuracy vs number of
   training samples, 90/10 split, showing bias-dominated behaviour.
2. **Confusion matrix** (report §5.3), 4,000 train / 1,000 test:
   ```
   [[448,  86],     row 0 = undamaged (534 images)
    [117, 349]]     row 1 = damaged   (466 images)
   ```
   → accuracy 79.7%, precision 0.802, recall 0.749 for the damaged class.
3. **Misclassified-image gallery** (report Fig. 5): false positives are tree
   branches, ridges, brick patterns, curtains; false negatives are cracks read
   as texture or pattern.
4. **Augmentation examples** (report Fig. 6) showing that shift/flip can
   produce physically invalid structures.
5. **γ tuning curve** for the InceptionV3-SVM hybrid (report Fig. 3).

---

## 5. The claims being tested

The paper's argument, which is what the reproduction must actually confirm:

1. Classical ML on raw pixels fails — all three land near the 54% majority-class
   baseline on validation despite 99%+ training accuracy.
2. Transfer learning with a frozen pretrained CNN works (~83%).
3. The frozen-CNN model is **bias**-limited; adding trainable parameters fixes
   bias but introduces **variance** (100% train / 74.2% val).
4. Augmentation trades training accuracy for a smaller generalisation gap.
5. Mid-network activations (layer 288) are better features for this domain than
   the top of the network, because ImageNet contains only 0.008% structural
   images.
