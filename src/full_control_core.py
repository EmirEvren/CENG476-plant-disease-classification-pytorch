from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from tqdm import tqdm

import official_data_setup
from efficientnet_models import create_efficientnet_b0_transfer
from transfer_models import create_resnet18_transfer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "PlantVillage"
ULTRA_MANIFEST = PROJECT_ROOT / "outputs" / "audit" / "official_leaf_safe_split_manifest_ultrastrict.csv"
ULTRA_SUMMARY = PROJECT_ROOT / "outputs" / "audit" / "official_leaf_safe_split_summary_ultrastrict.json"
EFF_CKPT = PROJECT_ROOT / "outputs" / "checkpoints" / "efficientnet_b0_official_ultrastrict_full_best.pt"
RES_CKPT = PROJECT_ROOT / "outputs" / "checkpoints" / "resnet18_official_ultrastrict_full_best.pt"
OUT = PROJECT_ROOT / "outputs" / "audit" / "full_control"
FIG = PROJECT_ROOT / "outputs" / "figures" / "full_control"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--bootstrap-samples", type=int, default=1000)
    return p.parse_args()


def activate():
    official_data_setup.MANIFEST_PATH = ULTRA_MANIFEST
    official_data_setup.SUMMARY_PATH = ULTRA_SUMMARY


def load_models(device, class_names):
    ec = torch.load(EFF_CKPT, map_location=device, weights_only=False)
    rc = torch.load(RES_CKPT, map_location=device, weights_only=False)
    for ckpt in (ec, rc):
        if list(ckpt.get("class_names", [])) != list(class_names):
            raise RuntimeError("Checkpoint/manifest class mapping mismatch.")
        if ckpt.get("test_used_during_training") is not False:
            raise RuntimeError("Checkpoint does not confirm locked-test training.")
    eff = create_efficientnet_b0_transfer(len(class_names), float(ec.get("dropout_rate", 0.3)), False).to(device)
    res = create_resnet18_transfer(len(class_names), float(rc.get("dropout_rate", 0.3)), False).to(device)
    eff.load_state_dict(ec["model_state_dict"], strict=True)
    res.load_state_dict(rc["model_state_dict"], strict=True)
    return eff.eval(), res.eval(), ec, rc


def collect(eff, res, loader, device):
    ys, pe, pr = [], [], []
    use_amp = device.type == "cuda"
    with torch.inference_mode():
        for x, y in tqdm(loader, desc="Locked test inference", unit="batch"):
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                le, lr = eff(x), res(x)
            ys.append(y.numpy())
            pe.append(torch.softmax(le.float(), 1).cpu().numpy())
            pr.append(torch.softmax(lr.float(), 1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(pe), np.concatenate(pr)


def metrics(y, p, n):
    pred = p.argmax(1)
    mp, mr, mf, _ = precision_recall_fscore_support(y, pred, labels=np.arange(n), average="macro", zero_division=0)
    _, _, wf, _ = precision_recall_fscore_support(y, pred, labels=np.arange(n), average="weighted", zero_division=0)
    return pred, float(accuracy_score(y, pred)), float(mp), float(mr), float(mf), float(wf)


def calibration(y, p, bins=15):
    pred = p.argmax(1)
    conf = p.max(1)
    correct = pred == y
    edges = np.linspace(0, 1, bins + 1)
    rows, ece = [], 0.0
    for i in range(bins):
        mask = (conf >= edges[i]) & (conf <= edges[i + 1] if i == bins - 1 else conf < edges[i + 1])
        count = int(mask.sum())
        if count:
            c = float(conf[mask].mean())
            a = float(correct[mask].mean())
            gap = abs(c - a)
            ece += count / len(y) * gap
        else:
            c = a = gap = np.nan
        rows.append({"bin": i, "lower": edges[i], "upper": edges[i+1], "count": count,
                     "mean_confidence": c, "accuracy": a, "gap": gap})
    true_p = np.clip(p[np.arange(len(y)), y], 1e-12, 1.0)
    nll = float(-np.log(true_p).mean())
    onehot = np.eye(p.shape[1], dtype=np.float32)[y]
    brier = float(np.mean(np.sum((p - onehot) ** 2, axis=1)))
    return float(ece), nll, brier, pd.DataFrame(rows)


def bootstrap(y, predictions, classes, samples):
    rng = np.random.default_rng(20260820)
    rows = []
    for name, pred in predictions.items():
        vals = np.empty((samples, 2), dtype=np.float64)
        for i in tqdm(range(samples), desc=f"Bootstrap {name}", unit="sample"):
            idx = rng.integers(0, len(y), size=len(y))
            cm = confusion_matrix(y[idx], pred[idx], labels=np.arange(classes))
            tp = np.diag(cm).astype(float)
            fp = cm.sum(0) - tp
            fn = cm.sum(1) - tp
            den = 2 * tp + fp + fn
            f1 = np.divide(2 * tp, den, out=np.zeros_like(tp), where=den > 0)
            vals[i, 0] = np.trace(cm) / cm.sum()
            vals[i, 1] = f1.mean()
        rows.append({
            "model": name,
            "accuracy_mean": vals[:,0].mean(),
            "accuracy_ci_low": np.percentile(vals[:,0], 2.5),
            "accuracy_ci_high": np.percentile(vals[:,0], 97.5),
            "macro_f1_mean": vals[:,1].mean(),
            "macro_f1_ci_low": np.percentile(vals[:,1], 2.5),
            "macro_f1_ci_high": np.percentile(vals[:,1], 97.5),
            "bootstrap_samples": samples,
        })
    return pd.DataFrame(rows)


def per_class(y, p, names, prefix):
    pred = p.argmax(1)
    labels = np.arange(len(names))
    prec, rec, f1, sup = precision_recall_fscore_support(y, pred, labels=labels, average=None, zero_division=0)
    cm = confusion_matrix(y, pred, labels=labels)
    frame = pd.DataFrame({
        "class_index": labels, "class_name": names, "precision": prec, "recall": rec, "f1": f1,
        "support": sup.astype(int), "errors": sup.astype(int) - np.diag(cm)
    })
    frame["error_rate"] = frame["errors"] / frame["support"].clip(lower=1)
    frame.sort_values(["f1", "support"]).to_csv(OUT / f"{prefix}_per_class.csv", index=False)
    pd.DataFrame(cm, index=names, columns=names).to_csv(OUT / f"{prefix}_confusion_matrix.csv")


def reliability_plot(table, path, title):
    t = table[table["count"] > 0]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0,1], [0,1], "--", label="Perfect")
    ax.plot(t["mean_confidence"], t["accuracy"], marker="o", label="Observed")
    ax.set(xlim=(0,1), ylim=(0,1), xlabel="Mean confidence", ylabel="Accuracy", title=title)
    ax.grid(alpha=.25); ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=220); plt.close(fig)


def error_sheet(frame, path, limit=48):
    frame = frame.head(limit)
    if frame.empty:
        return
    iw, ih, lh, cols = 180, 180, 82, 4
    rows = math.ceil(len(frame)/cols)
    canvas = Image.new("RGB", (iw*cols, (ih+lh)*rows), "white")
    draw = ImageDraw.Draw(canvas)
    for k, (_, row) in enumerate(frame.iterrows()):
        x0, y0 = (k % cols)*iw, (k // cols)*(ih+lh)
        with Image.open(DATASET_ROOT / row["relative_path"]) as im:
            im = im.convert("RGB"); im.thumbnail((iw, ih))
            slot = Image.new("RGB", (iw, ih), "white")
            slot.paste(im, ((iw-im.width)//2, (ih-im.height)//2))
        canvas.paste(slot, (x0, y0))
        draw.multiline_text((x0+2, y0+ih+2),
            f"conf={row['confidence']:.3f}\nT:{row['true_class'][:22]}\nP:{row['predicted_class'][:22]}",
            fill="black", spacing=2)
    canvas.save(path, quality=92)


def main():
    args = parse_args()
    if args.batch_size <= 0 or args.bootstrap_samples <= 0:
        raise ValueError("Positive arguments required.")
    OUT.mkdir(parents=True, exist_ok=True); FIG.mkdir(parents=True, exist_ok=True)
    activate()
    train_loader, val_loader, test_loader, names, split = official_data_setup.create_official_dataloaders(args.batch_size, 0)
    if len(test_loader.dataset) != 10709:
        raise RuntimeError("Locked test must contain 10,709 images.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eff, res, ec, rc = load_models(device, names)
    y, pe, pr = collect(eff, res, test_loader, device)
    pen = .5 * pe + .5 * pr
    paths = test_loader.dataset.frame["relative_path"].astype(str).tolist()
    np.savez_compressed(OUT / "locked_test_probabilities.npz", targets=y, efficientnet=pe, resnet18=pr, ensemble=pen)

    rows, preds = [], {}
    for name, p in [("EfficientNet-B0", pe), ("50/50 Ensemble", pen)]:
        prefix = "efficientnet" if name.startswith("Eff") else "ensemble"
        pred, acc, mp, mr, mf, wf = metrics(y, p, len(names))
        preds[name] = pred
        ece, nll, brier, cal = calibration(y, p)
        cal.to_csv(OUT / f"{prefix}_calibration_bins.csv", index=False)
        reliability_plot(cal, FIG / f"{prefix}_reliability.png", f"{name} reliability")
        per_class(y, p, names, prefix)
        conf = p.max(1)
        detail = pd.DataFrame({
            "relative_path": paths, "true_index": y, "true_class": [names[i] for i in y],
            "predicted_index": pred, "predicted_class": [names[i] for i in pred],
            "confidence": conf, "true_class_probability": p[np.arange(len(y)), y],
            "correct": pred == y
        })
        detail.to_csv(OUT / f"{prefix}_predictions.csv", index=False)
        errors = detail[~detail["correct"]].sort_values("confidence", ascending=False)
        errors.to_csv(OUT / f"{prefix}_errors.csv", index=False)
        error_sheet(errors, FIG / f"{prefix}_high_confidence_errors.jpg")
        rows.append({"model": name, "accuracy": acc, "macro_precision": mp, "macro_recall": mr,
                     "macro_f1": mf, "weighted_f1": wf, "errors": int((pred != y).sum()),
                     "ece_15_bins": ece, "nll": nll, "multiclass_brier": brier})
    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "metrics_calibration_summary.csv", index=False)
    ci = bootstrap(y, preds, len(names), args.bootstrap_samples)
    ci.to_csv(OUT / "bootstrap_95ci.csv", index=False)

    payload = {
        "protocol": "ultra_strict_dhash_quarantined_official_test",
        "train_images": len(train_loader.dataset), "validation_images": len(val_loader.dataset),
        "locked_test_images": len(test_loader.dataset), "test_set_modified": False,
        "test_used_for_training": False, "test_used_for_checkpoint_selection": False,
        "test_used_for_ensemble_weight_selection": False,
        "test_images_used_for_integrity_audit_only": True,
        "strict_dhash_pairs_after_quarantine": int(split.get("strict_dhash_pairs_after_quarantine", -1)),
        "efficientnet_checkpoint_epoch": int(ec["epoch"]), "resnet_checkpoint_epoch": int(rc["epoch"]),
        "metrics": rows, "bootstrap_95ci": ci.to_dict(orient="records")
    }
    with (OUT / "core_summary.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print("="*100)
    print("FULL CONTROL CORE")
    d = summary.copy(); d["accuracy"] *= 100
    print(d.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nBOOTSTRAP 95% CI")
    print(ci.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print("\nOutputs:", OUT)


if __name__ == "__main__":
    main()
