# Build It Yourself — a nine-module course

You are going to rebuild this project from nothing. Not by reading `repro/`,
but by hitting the same walls I hit and getting past them.

Every module ends with a **checkpoint**: a number you can compute yourself. If
your number matches, you got it right and you move on. If it doesn't, you have
a bug — and finding it is the module.

Several modules contain a **deliberate trap**. You will probably fall into it.
That is the design. The trap is flagged, but not explained, because the skill
being taught is *noticing that something is wrong when nothing has crashed*.

---

## Rules

1. **Do not open `repro/` until you have finished a module.** Then read the
   corresponding file and diff it against your thinking. Reading first turns a
   course into a transcription exercise.
2. **Keep a lab notebook.** One file, `practice/NOTES.md`. For every module
   write: what you expected, what happened, what you changed. This is where
   the learning actually gets stored, and it is also where your viva answers
   come from.
3. **Timebox.** If a module has you stuck for more than 45 minutes, use the
   hint ladder at the bottom of that module. Each rung gives away a little
   more. Take the smallest rung that unblocks you.
4. **Write every number down.** "It worked" is not a result. "Validation
   accuracy 0.796, majority baseline 0.534" is.

### Workspace

Work in `practice/`, which is git-ignored, so your code never collides with
mine:

```bash
mkdir -p practice
.venv/bin/python practice/your_script.py
```

Use the same virtual environment — rebuilding it is Module 0 only if you want
the practice.

---

## Module 0 — Environment

**Goal.** A Python environment where PyTorch can see your GPU.

**Brief.** Your system Python is 3.14; PyTorch does not publish wheels for it.
Create an isolated environment on a supported version and install PyTorch,
scikit-learn, matplotlib and Pillow. Nothing may be installed system-wide.

**Checkpoint.**

```python
import torch
torch.cuda.is_available()           # True
torch.cuda.get_device_name(0)       # 'NVIDIA GeForce RTX 4050 Laptop GPU'
```

**Why it matters.** "It works on my machine" is the most common reason a
reproduction fails. An isolated, pinned environment is the first deliverable
of any reproduction study, not an afterthought.

---

## Module 1 — Open the data

**Goal.** Load the PHI-Net Task 2 arrays and display a few images correctly.

**Brief.** The two zips are in `data/raw/`. Extract the `.npy` files, load
them, and plot a 3×4 grid of images with `matplotlib`.

> ⚠️ **Trap.** This module contains one. Nothing will crash. Your images may
> even look vaguely plausible. Before you move on, convince yourself the
> pixels you are plotting are the pixels the photographer captured.

**Checkpoint.** Once correct:

| Quantity | Value |
|---|---|
| `X_train` shape | `(11811, 224, 224, 3)` |
| `X_train` dtype | `float32` |
| min / max of the raw array | `-123.68` / `151.06` |
| min / max after your correction | `0.0` / `255.0` |
| are the corrected values integers? | yes, to within 1e-5 |

And the decisive check: **a photo of a brick wall must look like a brick
wall.** Orange brick, green foliage, blue sky.

**Why it matters.** Published datasets are often pre-processed in undocumented
ways. If you train on silently corrupted inputs, you get mediocre accuracy and
spend days blaming your model.

**Hint ladder.**
1. What is the range of an ordinary 8-bit image? What is the range of yours?
2. `-123.68` is not a random number. Search it.
3. Check the per-channel minima separately. Are they in the order you expect?

---

## Module 2 — Labels

**Goal.** Establish, with evidence, which numeric label means "damaged".

**Brief.** The labels are one-hot `(N, 2)` float arrays. Collapse them to
integers. Then determine which class is which — and *prove* it.

> ⚠️ **Trap.** The README inside the zip, the official task page, and the
> paper do not all agree. One of your three sources of truth is useless and
> one is ambiguous.

**Checkpoint.**

| Quantity | Value |
|---|---|
| raw pool class 0 count | 6,282 (53.2%) |
| raw pool class 1 count | 5,529 (46.8%) |
| test set | 745 / 715 |

Then answer in your notebook: **which raw label is "damaged", and what is your
evidence?**

**The hard part.** Try to design an automated check that confirms your answer —
something measurable that should differ between damaged and undamaged images.
Run it. Write down what it says *before* you look at any pictures.

**Why it matters.** A label flip is undetectable by accuracy: swap both class
names and every score is identical. Only the interpretation breaks — your
confusion matrix transposes, your "false positives" become false negatives,
and your recall-for-damage number means the opposite of what you say it does.
Some decisions cannot be validated by metrics at all.

**Hint ladder.**
1. The paper never states its convention directly. But it prints a confusion
   matrix, and it states the class balance. Can you combine them?
2. Whatever automated check you built — is it measuring damage, or is it
   measuring something that merely correlates with photo composition?
3. Look at twenty images with their labels printed above them.

---

## Module 3 — Splits

**Goal.** Reproduce the paper's exact train/validation split.

**Brief.** The paper used 5,913 images: 3,186 undamaged, 2,727 damaged, split
90/10. The modern release has 13,271 with the opposite class ratio. Subsample
to match, under a fixed seed, and save the splits to disk as `uint8`.

**Checkpoint.**

| Split | Undamaged | Damaged | Total |
|---|---|---|---|
| train | 2,870 | 2,451 | 5,321 |
| val | 316 | 276 | 592 |

**Two things to get right.** (a) The class ratio must be preserved — the paper
is explicit about this, and the reason is in Module 6. (b) The raw train array
is 7.1 GB of float32 and you have about 7 GB of free RAM. You must never hold
it all at once.

**Why it matters.** Reproducibility means someone else running your script
gets your numbers. A split that changes between runs makes every comparison
meaningless.

**Hint ladder.**
1. `np.load` takes an argument that avoids reading the whole file into memory.
2. Process in batches and convert dtype as you go.
3. `uint8` is lossless here and four times smaller. Why is it lossless?

---

## Module 4 — Classical baselines

**Goal.** Reproduce rows 1–3 of the paper's Table 1 — the models that fail.

**Brief.** Flatten each image to a 150,528-dim vector scaled to [0,1]. Train
k-nearest neighbours (k=5), logistic regression (L2, C=1.0) and an RBF SVM
(C=1.0, γ=0.001). Report train and validation accuracy for each.

> ⚠️ **Trap.** This module will probably take your machine down. When it does,
> do not just reduce the dataset size — work out *what specifically* consumed
> the memory. There are at least three distinct causes and they need different
> fixes.

**Checkpoint.** Within a point or two:

| Model | Train | Val |
|---|---|---|
| KNN (k=5) | 71.3% | 50.8% |
| Logistic regression | 99.8% | 54.0% |
| SVM (RBF), 2,700 train | 97.6% | 46.6% |

**The number that matters.** Compute the majority-class baseline on your
validation set before you interpret anything. Then answer: *are any of these
models better than guessing?*

**Go further.** Print the SVM's validation confusion matrix. It is not a
slightly-worse classifier. Describe in one sentence what it is actually doing.

**Why it matters.** 99.8% training accuracy is not a good result — it is a
diagnosis. With 28× more features than examples, memorisation is free. Every
model must be compared against a trivial baseline or its number means nothing.

**Hint ladder.**
1. How many bytes is a 5321 × 150528 float32 array? How much free RAM do you
   have? Now: does scikit-learn keep your dtype, or change it?
2. `n_jobs=-1` is not free. What does joblib do with your array?
3. KNN and the RBF SVM never need the raw features — only pairwise distances
   between them. What follows from that?

---

## Module 5 — Transfer learning

**Goal.** Reproduce the paper's headline model: ~80% validation accuracy.

**Brief.** Load InceptionV3 pretrained on ImageNet. Freeze every layer. Remove
its 1000-way classifier. Train a single `Linear(2048, 2)` with softmax on top,
using plain SGD at lr 0.01, batch 100, 4,000 iterations.

> ⚠️ **Trap.** The paper says it normalised inputs as `x/128 - 1`. Copying
> that formula into your code is the wrong move. Work out why before you run
> anything.

**Checkpoint.**

| | Train | Val |
|---|---|---|
| yours (224px input) | ~88% | ~79–80% |
| the paper's claim | 81.0% | 83.0% |

**The optimisation that makes everything else possible.** A naive loop runs
the CNN 4,000 times over your data. Think about what is actually changing
between iterations and what isn't. Done right, head training takes about
twenty seconds instead of hours — and Module 6 becomes feasible at all.

**Then investigate.** You will land below the paper's 83.0%. Two leads worth
chasing: (a) the paper prints a confusion matrix in its error-analysis
section — what accuracy does it imply, and how does that compare to yours?
(b) What input resolution was InceptionV3 trained at, and what are you giving
it?

**Why it matters.** Transfer learning is the single highest-leverage technique
for small image datasets. And "faithful to the paper" means faithful to the
*method*, not to a formula copied into a context where it no longer applies.

**Hint ladder.**
1. Preprocessing is not a universal constant — it is a property of the
   weights you loaded. Whose weights are you using?
2. The backbone is frozen. Does its output for a given image ever change?
3. Look up "bottleneck features" in the TensorFlow for Poets tutorial — the
   one the paper cites.

---

## Module 6 — Learning curve

**Goal.** Reproduce the paper's bias/variance diagnostic, then decide whether
you agree with it.

**Brief.** Retrain the head on 200, 400, 800 … 5,321 images (class ratio
preserved) and plot train and validation accuracy against training-set size.
Run three seeds and show the spread.

**Checkpoint.**

| Training images | Train | Val | Gap |
|---|---|---|---|
| 200 | 100.0% | 73.5% | +26.5 |
| 1,200 | 98.5% | 77.0% | +21.5 |
| 5,321 | 88.7% | 80.1% | +8.6 |

**The actual exercise.** The paper looked at its version of this plot and
wrote *"clearly a bias issue"*, then spent its next section adding model
capacity in response. Look at your plot. **Do you agree?** Write your answer
in your notebook, with reasons, before reading on anywhere.

Specifically: what does a large train/val gap indicate? What does a *flat*
validation curve indicate? What do you conclude when you see both at once —
a shrinking gap and a plateau?

**Why it matters.** This is the diagnostic that decides what you do next. Read
it as bias and you add parameters. Read it as variance and you add data or
regularisation. Read it wrong and you spend a week going the wrong way — which
is arguably what happened in the paper.

---

## Module 7 — Error analysis

**Goal.** Find out *how* your model fails, not just how often.

**Brief.** Produce a confusion matrix on validation and on the held-out test
set. Then render the most confident false positives and false negatives, with
the model's probability on each.

**Checkpoint.**

```
validation   [[255,  61],   79.6%
              [ 60, 216]]

held-out     [[581, 134],   81.2%
              [140, 605]]
```

**The actual exercise.** Look at the misclassified images and characterise the
failures in words. Not "the model is 80% accurate" but "the model fails on
*this kind* of image, because *this*".

Two questions to answer in your notebook:
- Among your false positives, how many are genuinely the model's fault?
- Is there a visual property shared by most of your false negatives? What does
  that tell you about what the model actually learned to detect?

**Why it matters.** Aggregate accuracy tells you nothing actionable. The error
structure tells you what to fix, what the dataset is missing, and what to say
when a panel asks why the number isn't higher. This is also where you discover
that some of your "errors" are the dataset's errors.

---

## Module 8 — Ship it

**Goal.** Turn results into deliverables.

**Brief.** A README that someone else can follow from a clean machine. A
two-page PDF covering problem statement, dataset, approach, implementation and
conclusions. A demo that classifies images in a couple of seconds with no
training and no network.

**Checkpoint.** Hand your repo to someone who has never seen it. Can they get
your headline number without asking you a question? That is the test.

**One principle worth adopting.** Generate the write-up *from* your results
files rather than typing numbers into a document. Numbers typed by hand go
stale the moment you rerun anything.

**Why it matters.** Four of the eight grading criteria are about
communication, not modelling. A perfect experiment nobody can run or
understand scores badly, and deservedly.

---

## If you want to go past the paper

These are open — I have not done them either:

- **MobileNetV1** and the **SVM on layer-288 activations** — the two Table 1
  rows still missing. The second tests a real hypothesis: ImageNet contains
  1,190 structural images out of 14.2 million, so are *mid*-network features
  better for this domain than the top of the network? Test several depths.
- **Fine-tuning.** Unfreeze the last block and retrain. You should be able to
  drive training accuracy to 100% and watch validation accuracy *fall*. Then
  fix it with augmentation and measure the trade.
- **Measure the label noise.** Sample 200 images, label them yourself blind,
  and compare with the dataset. That number is an upper bound on achievable
  accuracy, and nobody has published it.
- **Close-up versus wide shot.** Split the test set by whether the photo is a
  close-up, and report accuracy separately. The paper lists this as future
  work; your Module 7 results suggest it matters a lot.

---

## Suggested pace

| Day | Modules |
|---|---|
| 1 | 0–3 — environment and data. Expect this to take longer than you think. |
| 2 | 4–5 — baselines and transfer learning |
| 3 | 6–7 — diagnostics |
| 4 | 8 — deliverables |

Modules 1 and 2 are where the real lessons are. Do not rush them to get to the
neural network — the neural network is the easy part, and you now know it
trains in twenty seconds.
