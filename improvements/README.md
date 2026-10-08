# Improvements over the original paper

`repro/` reproduces Ho & Troncoso (CS229, 2018) at their own settings. This
folder is what we did **after** that — changes the paper did not make, each
measured separately on the same untouched 1,460-image Φ-Net test split.

```
faithful reproduction (paper's exact configuration)   81.2%
  + native 299px input, all 11,219 images             82.7%
  + modern backbone (EfficientNetV2) and an SVM       88.2%
  + ensemble of three modern backbones                88.5%
                                                      ──────
                                                      +7.3 points
```

Head to head against the paper's backbone, same test set, same pipeline:

| | InceptionV3 | ours | change |
|---|---|---|---|
| accuracy | 83.8% | **88.5%** | **+4.7** |
| ROC AUC | 0.914 | **0.943** | +0.029 |
| damaged recall | 81.2% | **87.0%** | +5.8 |
| damaged buildings missed | 140 | **97** | **−43** |
| false alarms | 134 | **94** | −40 |

Both error types fall at once, which means the features improved rather than
the operating point shifting.

---

## 1. `modern_backbones.py` — the feature extractor

The paper used InceptionV3, published in 2015, and MobileNetV1. Everything
since — EfficientNet (2019), Vision Transformers (2020), Swin (2021),
ConvNeXt (2022) — postdates their work.

We hold the data, split, classifier and every hyperparameter fixed, so the
**only** variable is the backbone. Each feature set gets both a linear softmax
head and an RBF SVM.

| Backbone | Year | Linear test | SVM test |
|---|---|---|---|
| InceptionV3 *(the paper's)* | 2015 | 82.3% | 83.8% |
| ResNet50 | 2015 | 83.1% | 84.5% |
| Swin Transformer (tiny) | 2021 | 85.0% | 85.9% |
| ConvNeXt (tiny) | 2022 | 84.7% | 86.6% |
| **EfficientNetV2** | 2021 | 85.7% | **88.2%** |

**Two findings.**

**(a) A modern backbone is worth +4.4 points** on identical data. The paper's
accuracy ceiling was partly an artefact of 2015 architecture.

**(b) The RBF SVM beats the linear head on every single backbone**, by 1.2 to
2.5 points. The paper tried an SVM on mid-network (layer-288) activations and
reached 75%; applied to top-layer features it is consistently the better
classifier.

### The methodological finding, which matters more

```
                 validation    test
ResNet50            0.867      0.845
EfficientNetV2      0.867      0.882
```

The two tie **exactly** on validation and differ by **3.7 points** on test. A
592-image validation set — the paper's split, which we inherited — cannot
distinguish them: one image is worth 0.17%, so a zero-point validation gap hid
a near-four-point test gap.

This is also why the validation-selected ensemble (87.5%) scores *below*
EfficientNetV2 alone (88.2%): validation picked the wrong members. The fix is
k-fold cross-validation or a substantially larger validation split.

```bash
python improvements/modern_backbones.py                 # ~20 min, extracts features
python improvements/modern_backbones.py --skip-extract  # ~1 min, reuses cache
```

---

## 2. `operating_point.py` — the decision threshold

The paper reports accuracy and nothing else. For the application it describes —
prioritising which buildings an engineer inspects after an earthquake — the two
errors are not equally costly:

| | consequence |
|---|---|
| false positive | an intact building is inspected → a wasted trip |
| false negative | a damaged building is cleared → someone walks into it |

Both the paper's model and ours use `argmax`, which silently fixes the
threshold at 0.5. Nobody chose that; it is simply what `argmax` does.

Choosing the threshold **on validation** to reach 90% recall, then reporting on
the untouched test set:

| | threshold 0.5 | threshold 0.319 |
|---|---|---|
| accuracy | 82.9% | 80.1% |
| recall (damaged) | 81.7% | **90.9%** |
| precision | 84.2% | 75.2% |
| **damaged buildings missed** | **136** | **68** |
| false alarms | 114 | 223 |

**Missed damage halves** — 136 to 68 — at a cost of 109 extra inspections, or
1.6 wasted trips per building saved. Accuracy *falls* by 2.8 points while the
system becomes substantially safer, which is precisely why accuracy is the
wrong figure of merit here.

Two threshold-independent measurements also come out of this:

- **ROC AUC 0.914** — how well the model *ranks* damaged above undamaged,
  independent of where the cut is placed. A far more robust summary than a
  single accuracy figure.
- **Expected calibration error 0.019** — when the model says 80%, it is right
  about 80% of the time. The probabilities can be acted on directly.

```bash
python improvements/operating_point.py --target-recall 0.90
```

Figure: `results/figures/07_operating_point.png`

---

## 3. `best_model.py` — the unconstrained best model

Nothing here is constrained by the paper. All 11,219 training images, modern
backbones, **test-time augmentation** (each image passed through the backbone
normally and horizontally mirrored, the two feature vectors averaged), and
model selection by **3-fold cross-validation over the training set** rather
than by the 592-image validation split.

The cross-validation was the point: section 1 showed validation cannot
distinguish two backbones that differ by 3.7 points on test. Cross-validation
uses thousands of held-out images per decision instead of 592.

**It did not fix the problem.** The CV ranking was exactly backwards:

| Backbone | CV accuracy | Test accuracy |
|---|---|---|
| Swin-T | 0.8574 ← CV winner | 86.9% ← worst |
| ConvNeXt-T | 0.8552 | 87.0% |
| EfficientNetV2 | 0.8534 ← CV last | 88.2% ← best |

The CV spread is 0.4 points against a fold-to-fold standard deviation of
0.3–0.6, so the three are statistically indistinguishable and the ordering is
noise. This is an honest negative result: **on a dataset this size you cannot
reliably select the best backbone.**

The answer is not to select at all. Averaging all three gives **88.5%**, above
every member. Individually each is wrong on 173–236 of the 1,460 test images;
all three are wrong on only **80**. They fail on different photographs, which
is exactly the condition that makes ensembling work — and the opposite of what
we found when ensembling two classifiers over the *same* features, where 188
of 250 errors were shared and the ensemble gained nothing.

```bash
python improvements/best_model.py                # ~40 min, extracts features
python improvements/best_model.py --skip-extract # reuses the cache
```

---

## 4. Improvements implemented inside `repro/`

Two further changes share the reproduction pipeline too closely to separate,
so they live in `repro/train_full.py`:

**Input resolution, +2.2 points.** The paper fed 224×224 images into a network
pretrained at 299×299. That shrinks the deepest feature grid from 8×8 to 5×5,
so the 2,048 features are pooled over 25 spatial positions instead of 64 — a
hairline crack in one cell of a 5×5 grid is averaged away with a fifth of the
image. Upscaling adds no detail; it presents the same information at the scale
the network was built to read.

**Training-set size, +1.3 points.** The Φ-Net release has more than doubled
since 2018 (5,913 → 13,271 images). Training on all 11,219 available images
gains 1.3 ± 0.5 points and **saturates near 8,000**, which establishes that
data volume is no longer the binding constraint.

---

## 5. Analyses with no counterpart in the paper

**Bias/variance with error bars** (`repro/learning_curve.py`). The paper read
its learning curve as "clearly a bias issue" and responded by adding capacity.
Averaged over three seeds we measure a train/validation gap of 26 points at 200
images falling to 8.6 at 5,321 — the signature of **variance**, not bias — with
validation flattening near 80%.

**Ensemble error-correlation.** Ensembling only helps when members fail on
*different* images. The linear head and the SVM on InceptionV3 features share
188 of 250 errors, so averaging them changes nothing. That is direct evidence
the **representation**, not the classifier, is the limit — and the argument for
fine-tuning next.

**A real held-out test set.** The paper kept none: *"we decided not to set
aside images for testing."* Every number it reports is validation accuracy on a
set it also used to make modelling decisions.

---

## 6. What we would do next

1. **k-fold cross-validation** for model selection, so the backbone choice is
   trustworthy rather than a coin flip between two tied candidates.
2. **EfficientNetV2 on all 11,219 images** — the comparison above used the
   paper-matched 5,321, so there is likely more available.
3. **Fine-tune the best backbone** at a learning rate around 1e-4 (roughly 1%
   of a typical weight per step, against 100% at the paper's 0.01), with
   augmentation and early stopping to control the resulting overfitting.
4. **Measure the label-noise rate.** We found mislabelled images by inspection;
   nobody has published the rate for Φ-Net, and it is the true ceiling.
