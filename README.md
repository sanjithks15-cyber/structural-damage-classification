**# Structural Damage Image Classification**

Binary classification of structural photographs as **\*\*damaged\*\*** or **\*\*undamaged\*\***,

reproducing and extending Ho & Troncoso, \*Structural Damage Image

Classification\*, CS229 Final Project, Stanford 2018.

Machine Learning (UE24CS352A) mini-project.

Results, method and conclusions are in [\`docs/writeup.pdf\`]\(docs/writeup.pdf).

**---**

**## 1. Prerequisites**

\| | |

\|---|---|

\| Python | 3.12 (3.13+ has no PyTorch wheels yet) |

\| GPU | NVIDIA, \~6 GB VRAM — tested on an RTX 4050 laptop |

\| Disk | \~12 GB free, or \~20 GB to build the production model |

\| RAM | 8 GB free |

Everything runs on CPU without a GPU, but feature extraction takes minutes

rather than seconds.

**---**

**## 2. Get the dataset**

PEER Hub ImageNet (Φ-Net), **\*\*Task 2 — Damage State\*\***, CC BY-NC-SA 4.0:

<[https://apps.peer.berkeley.edu/phi-net/](https://apps.peer.berkeley.edu/phi-net/)>. Not redistributed here.

\`\`\`bash

mkdir -p data/raw

curl -L -o data/raw/phi_task2_1.zip "[https://apps.peer.berkeley.edu/phichallenge/dataset/kfs55dd33](https://apps.peer.berkeley.edu/phichallenge/dataset/kfs55dd33)"

curl -L -o data/raw/phi_task2_2.zip "[https://apps.peer.berkeley.edu/phichallenge/dataset/task2_damage_state_2.zip](https://apps.peer.berkeley.edu/phichallenge/dataset/task2_damage_state_2.zip)"

\`\`\`

**\*\*Check the sizes before continuing\*\*** — a truncated or duplicated download

produces a zip that still opens in some tools:

\`\`\`bash

stat -c '%s %n' data/raw/\*.zip

\#   175290492 data/raw/phi_task2_1.zip

\#  1410780428 data/raw/phi_task2_2.zip

\`\`\`

The distributed arrays are BGR with the ImageNet mean pre-subtracted, and their

label convention is inverted relative to the paper's. \`prepare_data.py\` undoes

both; \`verify_labels.py\` and \`raw_label_check.py\` check the result.

**---**

**## 3. Set up the environment**

\`\`\`bash

curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

uv venv --python 3.12 .venv

uv pip install -r requirements.txt

\`\`\`

Nothing is installed system-wide; deleting \`.venv/\` undoes all of it.

**---**

**## 4. Run**

In order from the repository root — step 1 writes files the rest depend on.

\`\`\`bash

\# 1. Build the train/val/test splits

.venv/bin/python repro/prepare_data.py

\# 2. Verify the data and the GPU maths before trusting any model

.venv/bin/python repro/verify_labels.py

.venv/bin/python repro/raw_label_check.py

.venv/bin/python repro/baselines.py --verify

\# 3. Classical baselines

.venv/bin/python repro/baselines.py --model all

\# 4. Transfer learning (reproduction)

.venv/bin/python repro/inception_transfer.py --extract --train

.venv/bin/python repro/inception_transfer.py --extract --train --size 299

\# 5. Diagnostics

.venv/bin/python repro/learning_curve.py --seeds 3

.venv/bin/python repro/error_analysis.py

\# 6. Production model — all available data at native resolution

.venv/bin/python repro/extract_pool_features.py --size 299

.venv/bin/python repro/train_full.py --size 299 --seeds 3

# 7. Regenerate the write-up from the results

.venv/bin/python repro/make_writeup.py

### Windows

For Windows, use the Windows-compatible baseline verification script:

```powershell
python windows\baselines_windows.py --verify

\`\`\`

Approximate runtimes: step 1 \~3 min, step 3 \~4 min, step 4 \~30 s,

step 5 \~6 min, step 6 \~6 min.

**### Live demo**

\`\`\`bash

.venv/bin/python repro/demo.py                     # classify 8 unseen images

.venv/bin/python repro/demo.py --score             # metrics on the test set

.venv/bin/python repro/demo.py --errors            # the model's worst mistakes

.venv/bin/python repro/demo.py --image photo.jpg   # any image file

\`\`\`

\`--show\` opens the figure, \`--seed N\` repeats a particular sample, \`--repro\`

uses the paper-matched model instead of the production one.

**---**

**## 5. What each file does**

**### \`repro/\` — all analysis code, written for this study**

\| File | Purpose |

\|---|---|

\| \`prepare_data.py\` | builds the splits; undoes the BGR mean-subtraction and the label inversion |

\| \`verify_labels.py\` | checks the label polarity three ways |

\| \`raw_label_check.py\` | shows raw labels verbatim; writes the source-index mapping |

\| \`baselines.py\` | KNN, logistic regression, RBF SVM (\`--verify\` checks the GPU maths against scikit-learn) |

\| \`inception_transfer.py\` | frozen InceptionV3 + linear head — the paper's model |

\| \`extract_pool_features.py\` | features for all 11,811 pool images |

\| \`train_full.py\` | **\*\*production model\*\*** — all data, native resolution |

\| \`learning_curve.py\` | bias/variance diagnostic |

\| \`error_analysis.py\` | confusion matrices and misclassified-image galleries |

\| \`export_images.py\` | dataset → browsable JPEGs |

\| \`make_writeup.py\` | generates \`docs/writeup.pdf\` from \`results/\*.json\` |

\| \`demo.py\` | live demonstration |

**### Everything else**

\| Path | Contents |

\|---|---|

\| \`docs/writeup.pdf\` | the two-page write-up — **\*\*start here for results\*\*** |

\| \`results/\*.json\` | every metric reported, written by the scripts above |

\| \`results/head\_\*.pt\` | the trained classifiers (18 KB each) — what \`demo.py\` loads |

\| \`data/\` | dataset — git-ignored, created by \`prepare_data.py\` |

**---**

**## 6. Troubleshooting**

\| Symptom | Cause |

\|---|---|

\| \`Missing [...] in data/raw/extracted\` | dataset zips not in \`data/raw/\` |

\| \`BadZipFile\` | truncated download — recheck the byte counts in §2 |

\| \`incv3\_\*.npy missing\` | run \`inception_transfer.py --extract\` before \`--train\` |

\| \`head_full_299.pt not found\` | run \`train_full.py\`, or pass \`--repro\` to the demo |

\| \`CUDA out of memory\` | close other GPU programs; the baselines need \~4 GB VRAM |

\| Machine swaps during the baselines | expected under 8 GB free RAM |

**---**

**## License**

Analysis code: MIT. The Φ-Net dataset is CC BY-NC-SA 4.0 and is not

redistributed here.   