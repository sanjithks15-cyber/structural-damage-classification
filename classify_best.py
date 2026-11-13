"""
Best model on your own photos: shows each of the 3 models' probabilities,
then the averaged ensemble probability and the final classification.

    .venv/bin/python classify_best.py demo_images
    .venv/bin/python classify_best.py a.jpg b.png --show

Needs the cached SVMs in results/best_model/ (built by run_best.sh /
improvements/best_model.py, then improvements/demo_best.py).
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "improvements"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn.functional as F
from joblib import load as jload
from PIL import Image

from demo_best import BACKBONES, BACKBONE_CFG, CLASSES, FEAT, FIGS

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
LABEL = {"efficientnetv2": "EfficientNetV2", "convnext_tiny": "ConvNeXt",
         "swin_tiny": "Swin"}


def collect(paths):
    files = []
    for p in map(Path, paths):
        if p.is_dir():
            files += sorted(f for f in p.iterdir() if f.suffix.lower() in EXTS)
        elif p.is_file():
            files.append(p)
    return files


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("images", nargs="+")
    ap.add_argument("--show", action="store_true")
    a = ap.parse_args()

    files = collect(a.images)
    if not files:
        sys.exit("no images found")
    svms = {}
    for n in BACKBONES:
        p = FEAT / f"svm_{n}.joblib"
        if not p.exists():
            sys.exit(f"{p} missing - run ./run_best.sh first")
        svms[n] = jload(p)

    imgs = np.stack([np.asarray(Image.open(f).convert("RGB")
                                .resize((224, 224), Image.BILINEAR),
                                dtype=np.uint8) for f in files])
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    per = {}
    for n in BACKBONES:
        timm_id, size = BACKBONE_CFG[n]
        net = timm.create_model(timm_id, pretrained=True, num_classes=0)
        net.eval().to(dev)
        cfg = timm.data.resolve_data_config({}, model=net)
        mean = torch.tensor(cfg["mean"]).view(1, 3, 1, 1).to(dev)
        std = torch.tensor(cfg["std"]).view(1, 3, 1, 1).to(dev)
        with torch.no_grad():
            x = torch.from_numpy(imgs).to(dev).permute(0, 3, 1, 2).float() / 255
            if size != 224:
                x = F.interpolate(x, size=(size, size), mode="bilinear",
                                  align_corners=False)
            x = (x - mean) / std
            f = (net(x) + net(torch.flip(x, dims=[3]))) / 2      # TTA
        per[n] = svms[n].predict_proba(f.cpu().numpy())[:, 1]    # P(damaged)
        del net
        if dev == "cuda":
            torch.cuda.empty_cache()

    ens = np.mean([per[n] for n in BACKBONES], axis=0)

    print("\n  P(damaged) from each model, then the ensemble (their average)\n")
    head = f"  {'image':<18}" + "".join(f"{LABEL[n]:>15}" for n in BACKBONES)
    print(head + f"{'ENSEMBLE':>12}   FINAL")
    print("  " + "-" * (len(head) - 2 + 24))
    titles = []
    for i, f in enumerate(files):
        dmg = ens[i] >= 0.5
        verdict = CLASSES[int(dmg)].upper()
        row = f"  {f.name[:18]:<18}" + "".join(f"{per[n][i]:>15.1%}"
                                                for n in BACKBONES)
        print(row + f"{ens[i]:>12.1%}   {verdict}")
        titles.append((f"{f.name[:22]}\n{verdict}  P(dam)={ens[i]:.0%}",
                       "#b3512f" if dmg else "#2f7d32"))

    cols = min(5, len(files))
    rows = int(np.ceil(len(files) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 3.6),
                             squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, im, t in zip(axes.ravel(), imgs, titles):
        ax.imshow(im)
        ax.set_title(t[0], fontsize=9, color=t[1])
    fig.suptitle("Ensemble of EfficientNetV2 + ConvNeXt + Swin", fontsize=12)
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    out = FIGS / "classify_best.png"
    fig.savefig(out, dpi=95)
    print(f"\n  figure: {out}")
    if a.show:
        import subprocess
        subprocess.Popen(["xdg-open", str(out)], stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)


if __name__ == "__main__":
    main()
