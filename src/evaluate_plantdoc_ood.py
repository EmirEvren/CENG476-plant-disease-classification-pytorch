from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from data_setup import evaluation_transform
from efficientnet_models import create_efficientnet_b0_transfer
from transfer_models import create_resnet18_transfer
from full_control_core import EFF_CKPT, RES_CKPT, OUT

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PLANTDOC = PROJECT_ROOT / "data" / "external" / "PlantDoc-Dataset" / "test"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PLANTDOC_TO_PLANTVILLAGE = {
    "Apple Scab Leaf": "Apple___Apple_scab",
    "Apple leaf": "Apple___healthy",
    "Apple rust leaf": "Apple___Cedar_apple_rust",
    "Bell_pepper leaf spot": "Pepper,_bell___Bacterial_spot",
    "Bell_pepper leaf": "Pepper,_bell___healthy",
    "Blueberry leaf": "Blueberry___healthy",
    "Cherry leaf": "Cherry_(including_sour)___healthy",
    "Corn Gray leaf spot": "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn leaf blight": "Corn_(maize)___Northern_Leaf_Blight",
    "Corn rust leaf": "Corn_(maize)___Common_rust_",
    "Peach leaf": "Peach___healthy",
    "Potato leaf early blight": "Potato___Early_blight",
    "Potato leaf late blight": "Potato___Late_blight",
    "Raspberry leaf": "Raspberry___healthy",
    "Soyabean leaf": "Soybean___healthy",
    "Squash Powdery mildew leaf": "Squash___Powdery_mildew",
    "Strawberry leaf": "Strawberry___healthy",
    "Tomato Early blight leaf": "Tomato___Early_blight",
    "Tomato Septoria leaf spot": "Tomato___Septoria_leaf_spot",
    "Tomato leaf bacterial spot": "Tomato___Bacterial_spot",
    "Tomato leaf late blight": "Tomato___Late_blight",
    "Tomato leaf mosaic virus": "Tomato___Tomato_mosaic_virus",
    "Tomato leaf yellow virus": "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato leaf": "Tomato___healthy",
    "Tomato mold leaf": "Tomato___Leaf_Mold",
    "grape leaf black rot": "Grape___Black_rot",
    "grape leaf": "Grape___healthy",
}


class PlantDocMappedDataset(Dataset):
    def __init__(self, root, class_names):
        self.root = Path(root)
        self.class_names = class_names
        lookup = {name: i for i, name in enumerate(class_names)}
        rows = []
        for source, target in PLANTDOC_TO_PLANTVILLAGE.items():
            folder = self.root / source
            if not folder.is_dir():
                continue
            if target not in lookup:
                raise RuntimeError(f"PlantVillage target missing from checkpoint: {target}")
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append({
                        "path": path,
                        "source_class": source,
                        "target_class": target,
                        "target_index": lookup[target],
                    })
        if not rows:
            raise RuntimeError(f"No mapped PlantDoc images found under {self.root}")
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        with Image.open(row["path"]) as image:
            x = evaluation_transform(image.convert("RGB"))
        return x, int(row["target_index"]), str(row["path"]), row["source_class"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--plantdoc-test", type=Path, default=DEFAULT_PLANTDOC)
    p.add_argument("--batch-size", type=int, default=32)
    return p.parse_args()


def load_models(device):
    ec = torch.load(EFF_CKPT, map_location=device, weights_only=False)
    rc = torch.load(RES_CKPT, map_location=device, weights_only=False)
    names = list(ec["class_names"])
    if list(rc["class_names"]) != names:
        raise RuntimeError("ResNet/EfficientNet class mappings differ.")
    eff = create_efficientnet_b0_transfer(len(names), float(ec.get("dropout_rate", .3)), False).to(device)
    res = create_resnet18_transfer(len(names), float(rc.get("dropout_rate", .3)), False).to(device)
    eff.load_state_dict(ec["model_state_dict"], strict=True)
    res.load_state_dict(rc["model_state_dict"], strict=True)
    return eff.eval(), res.eval(), names


def collect(eff, res, loader, device):
    ys, pe, pr, paths, source_classes = [], [], [], [], []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for x, y, batch_paths, batch_sources in tqdm(loader, desc="PlantDoc OOD", unit="batch"):
            x = x.to(device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                le, lr = eff(x), res(x)
            ys.append(y.numpy())
            pe.append(torch.softmax(le.float(), 1).cpu().numpy())
            pr.append(torch.softmax(lr.float(), 1).cpu().numpy())
            paths.extend(batch_paths)
            source_classes.extend(batch_sources)
    return np.concatenate(ys), np.concatenate(pe), np.concatenate(pr), paths, source_classes


def report(name, probs, y, names, paths, sources):
    pred = probs.argmax(1)
    acc = float(accuracy_score(y, pred))
    macro = float(f1_score(y, pred, labels=sorted(set(y.tolist())), average="macro", zero_division=0))
    detail = pd.DataFrame({
        "path": paths,
        "plantdoc_source_class": sources,
        "mapped_true_index": y,
        "mapped_true_class": [names[i] for i in y],
        "predicted_index": pred,
        "predicted_class": [names[i] for i in pred],
        "confidence": probs.max(1),
        "correct": pred == y,
    })
    prefix = "efficientnet" if name.startswith("Eff") else "ensemble"
    detail.to_csv(OUT / f"plantdoc_{prefix}_predictions.csv", index=False)
    rows = []
    for source in sorted(detail["plantdoc_source_class"].unique()):
        part = detail[detail["plantdoc_source_class"] == source]
        rows.append({
            "plantdoc_source_class": source,
            "mapped_target_class": part.iloc[0]["mapped_true_class"],
            "support": len(part),
            "accuracy": float(part["correct"].mean()),
            "errors": int((~part["correct"]).sum()),
        })
    pd.DataFrame(rows).to_csv(OUT / f"plantdoc_{prefix}_per_class.csv", index=False)
    return {"model": name, "images": len(y), "accuracy": acc, "mapped_macro_f1": macro, "errors": int((pred != y).sum())}


def main():
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if not args.plantdoc_test.is_dir():
        raise FileNotFoundError(
            f"PlantDoc test directory not found: {args.plantdoc_test}\n"
            "Clone the official PlantDoc-Dataset repository first."
        )
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eff, res, names = load_models(device)
    ds = PlantDocMappedDataset(args.plantdoc_test, names)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    y, pe, pr, paths, sources = collect(eff, res, loader, device)
    ensemble = .5*pe + .5*pr

    mapping_rows = [{"plantdoc_class": k, "plantvillage_class": v,
                     "folder_present": bool((args.plantdoc_test / k).is_dir())}
                    for k, v in PLANTDOC_TO_PLANTVILLAGE.items()]
    pd.DataFrame(mapping_rows).to_csv(OUT / "plantdoc_label_mapping.csv", index=False)

    results = [
        report("EfficientNet-B0", pe, y, names, paths, sources),
        report("50/50 Ensemble", ensemble, y, names, paths, sources),
    ]
    payload = {
        "dataset": "Cropped-PlantDoc test split",
        "dataset_path": str(args.plantdoc_test.resolve()),
        "purpose": "External out-of-domain generalization probe; never used for training or tuning.",
        "mapping_is_manual_cross_dataset_semantic_mapping": True,
        "mapped_folders_present": int(sum(row["folder_present"] for row in mapping_rows)),
        "mapped_images": len(ds),
        "results": results,
        "important_limit": (
            "PlantDoc and PlantVillage class definitions and image distributions are not identical. "
            "Treat this as an OOD stress test, not a directly comparable replacement benchmark."
        ),
    }
    with (OUT / "plantdoc_ood_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("="*88)
    print("PLANTDOC EXTERNAL / OOD TEST")
    print("="*88)
    for row in results:
        print(f"{row['model']}: accuracy={row['accuracy']*100:.2f}% | mapped Macro-F1={row['mapped_macro_f1']:.4f} | errors={row['errors']}/{row['images']}")
    print("Mapped images:", len(ds))
    print("Output:", OUT / "plantdoc_ood_summary.json")


if __name__ == "__main__":
    main()
