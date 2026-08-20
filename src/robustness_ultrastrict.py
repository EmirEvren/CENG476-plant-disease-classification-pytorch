from __future__ import annotations

import argparse
import io
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

import official_data_setup
from data_setup import evaluation_transform
from full_control_core import DATASET_ROOT, FIG, OUT, activate, collect, load_models


class StressDataset(Dataset):
    def __init__(self, frame, perturbation):
        self.frame = frame.reset_index(drop=True)
        self.perturbation = perturbation

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(DATASET_ROOT / str(row["relative_path"])) as image:
            image = image.convert("RGB")
            if self.perturbation is not None:
                image = self.perturbation(image)
            tensor = evaluation_transform(image)
        return tensor, int(row["class_index"])


def center_occlude(image, fraction=0.60):
    result = image.copy()
    w, h = result.size
    bw, bh = int(w*fraction), int(h*fraction)
    left, top = (w-bw)//2, (h-bh)//2
    ImageDraw.Draw(result).rectangle((left, top, left+bw, top+bh), fill=(128,128,128))
    return result


def border_occlude(image, keep=0.60):
    w, h = image.size
    kw, kh = int(w*keep), int(h*keep)
    left, top = (w-kw)//2, (h-kh)//2
    right, bottom = left+kw, top+kh
    result = Image.new("RGB", image.size, (128,128,128))
    result.paste(image.crop((left, top, right, bottom)), (left, top))
    return result


def jpeg(image, quality=30):
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=quality)
    buffer.seek(0)
    with Image.open(buffer) as loaded:
        return loaded.convert("RGB").copy()


def conditions():
    return {
        "clean": None,
        "brightness_0.60": lambda im: ImageEnhance.Brightness(im).enhance(0.60),
        "brightness_1.40": lambda im: ImageEnhance.Brightness(im).enhance(1.40),
        "contrast_0.60": lambda im: ImageEnhance.Contrast(im).enhance(0.60),
        "gaussian_blur_radius_2": lambda im: im.filter(ImageFilter.GaussianBlur(2.0)),
        "jpeg_quality_30": lambda im: jpeg(im, 30),
        "rotation_15deg": lambda im: im.rotate(
            15, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=(128,128,128)
        ),
        "center_occluded_60pct": lambda im: center_occlude(im, 0.60),
        "border_occluded_keep_center_60pct": lambda im: border_occlude(im, 0.60),
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    activate()
    _, _, test_loader, names, _ = official_data_setup.create_official_dataloaders(args.batch_size, 0)
    frame = test_loader.dataset.frame.reset_index(drop=True)
    if len(frame) != 10709:
        raise RuntimeError("Locked test must contain 10,709 images.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eff, res, _, _ = load_models(device, names)
    rows = []

    for condition, perturb in conditions().items():
        loader = DataLoader(
            StressDataset(frame, perturb),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        y, pe, pr = collect(eff, res, loader, device)
        ensemble = 0.5 * pe + 0.5 * pr
        for model_name, probs in [
            ("EfficientNet-B0", pe),
            ("50/50 Ensemble", ensemble),
        ]:
            pred = probs.argmax(1)
            rows.append({
                "condition": condition,
                "model": model_name,
                "accuracy": float(accuracy_score(y, pred)),
                "macro_f1": float(f1_score(y, pred, labels=np.arange(len(names)), average="macro", zero_division=0)),
                "errors": int((pred != y).sum()),
            })

    result = pd.DataFrame(rows)
    clean = result[result["condition"] == "clean"].set_index("model")["accuracy"].to_dict()
    result["accuracy_drop_pp"] = result.apply(
        lambda row: (clean[row["model"]] - row["accuracy"]) * 100.0,
        axis=1,
    )
    result.to_csv(OUT / "robustness_stress.csv", index=False)

    shortcut = result[result["condition"].isin([
        "clean", "center_occluded_60pct", "border_occluded_keep_center_60pct"
    ])].copy()
    shortcut.to_csv(OUT / "shortcut_occlusion_stress.csv", index=False)

    for model_name in result["model"].unique():
        part = result[result["model"] == model_name]
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(part["condition"], part["accuracy"] * 100.0)
        ax.set_ylabel("Accuracy (%)")
        ax.set_title(f"{model_name} - robustness / shortcut stress")
        ax.tick_params(axis="x", rotation=35)
        ax.set_ylim(0, 100.5)
        ax.grid(axis="y", alpha=.25)
        fig.tight_layout()
        slug = "efficientnet" if model_name.startswith("Eff") else "ensemble"
        fig.savefig(FIG / f"{slug}_robustness_stress.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    payload = {
        "locked_test_images": 10709,
        "used_for_tuning": False,
        "interpretation": (
            "These are post-hoc robustness and shortcut stress tests. "
            "Center/border occlusion is not a segmentation proof; compare the "
            "relative degradation as qualitative shortcut evidence."
        ),
        "rows": result.to_dict(orient="records"),
    }
    with (OUT / "robustness_stress_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    display = result.copy()
    display["accuracy"] *= 100
    print("="*96)
    print("ROBUSTNESS / SHORTCUT STRESS")
    print("="*96)
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nOutputs:", OUT)


if __name__ == "__main__":
    main()
