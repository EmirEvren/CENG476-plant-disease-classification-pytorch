from __future__ import annotations

import argparse
import json
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

import official_data_setup
from data_setup import evaluation_transform
from efficientnet_models import create_efficientnet_b0_transfer
from full_control_core import DATASET_ROOT, OUT, activate

SEED = 20260820


class FrameDataset(Dataset):
    def __init__(self, frame, labels):
        self.frame = frame.reset_index(drop=True)
        self.labels = np.asarray(labels, dtype=np.int64)

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        with Image.open(DATASET_ROOT / str(row["relative_path"])) as image:
            x = evaluation_transform(image.convert("RGB"))
        return x, int(self.labels[index])


def balanced_sample(frame, per_class, seed):
    pieces = []
    for _, group in frame.groupby("class_index", sort=True):
        if len(group) < per_class:
            raise RuntimeError("Not enough images for random-label sanity sample.")
        pieces.append(group.sample(n=per_class, random_state=seed))
    return pd.concat(pieces, ignore_index=True)


def evaluate(model, loader, device, classes):
    ys, ps = [], []
    model.eval()
    with torch.inference_mode():
        for x, y in loader:
            logits = model(x.to(device))
            ys.append(y.numpy())
            ps.append(logits.argmax(1).cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return float(accuracy_score(y, p)), float(f1_score(y, p, labels=np.arange(classes), average="macro", zero_division=0))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--train-per-class", type=int, default=20)
    p.add_argument("--validation-per-class", type=int, default=10)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    if min(args.train_per_class, args.validation_per_class, args.epochs, args.batch_size) <= 0:
        raise ValueError("All arguments must be positive.")
    OUT.mkdir(parents=True, exist_ok=True)
    activate()
    frame, class_names, _ = official_data_setup._load_manifest()
    train_frame = balanced_sample(frame[frame["split"] == "train"], args.train_per_class, SEED)
    val_frame = balanced_sample(frame[frame["split"] == "validation"], args.validation_per_class, SEED + 1)

    true_train_labels = train_frame["class_index"].astype(int).to_numpy()
    rng = np.random.default_rng(SEED)
    random_train_labels = true_train_labels.copy()
    rng.shuffle(random_train_labels)
    unchanged = int((random_train_labels == true_train_labels).sum())

    train_ds = FrameDataset(train_frame, random_train_labels)
    val_ds = FrameDataset(val_frame, val_frame["class_index"].astype(int).to_numpy())
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=generator, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_efficientnet_b0_transfer(len(class_names), dropout_rate=0.3, pretrained=True).to(device)
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("classifier.")
    optimizer = AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    history = []
    for epoch in range(1, args.epochs + 1):
        model.eval()
        model.classifier.train()
        running, seen, correct = 0.0, 0, 0
        for x, y in tqdm(train_loader, desc=f"Random-label epoch {epoch}/{args.epochs}", unit="batch"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * len(y)
            seen += len(y)
            correct += int((logits.argmax(1) == y).sum().item())
        val_acc, val_f1 = evaluate(model, val_loader, device, len(class_names))
        history.append({
            "epoch": epoch,
            "random_label_train_loss": running/seen,
            "random_label_train_accuracy": correct/seen,
            "true_label_validation_accuracy": val_acc,
            "true_label_validation_macro_f1": val_f1,
        })
        print(history[-1])

    pd.DataFrame(history).to_csv(OUT / "random_label_sanity_history.csv", index=False)
    final = history[-1]
    passed = bool(final["true_label_validation_accuracy"] < 0.10 and final["true_label_validation_macro_f1"] < 0.10)
    payload = {
        "purpose": "Pipeline/label-leak sanity control; training labels were shuffled.",
        "classes": len(class_names),
        "chance_accuracy": 1.0 / len(class_names),
        "train_images": len(train_ds),
        "validation_images": len(val_ds),
        "random_labels_unchanged_by_chance": unchanged,
        "epochs": args.epochs,
        "backbone_frozen": True,
        "validation_uses_true_labels": True,
        "final_validation_accuracy": final["true_label_validation_accuracy"],
        "final_validation_macro_f1": final["true_label_validation_macro_f1"],
        "pass_threshold": 0.10,
        "pass": passed,
        "note": "This is a sanity check, not a formal statistical proof of zero leakage."
    }
    with (OUT / "random_label_sanity_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("="*80)
    print("RANDOM-LABEL SANITY:", "PASS" if passed else "REVIEW")
    print(f"Chance accuracy: {payload['chance_accuracy']*100:.2f}%")
    print(f"True-label validation accuracy after random-label training: {final['true_label_validation_accuracy']*100:.2f}%")
    print(f"Macro-F1: {final['true_label_validation_macro_f1']:.4f}")


if __name__ == "__main__":
    main()
