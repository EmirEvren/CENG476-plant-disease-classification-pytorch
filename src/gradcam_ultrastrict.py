from __future__ import annotations

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from tqdm import tqdm

from data_setup import evaluation_transform
from efficientnet_models import create_efficientnet_b0_transfer
from full_control_core import DATASET_ROOT, EFF_CKPT, FIG, OUT


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--images-per-sheet", type=int, default=12)
    return p.parse_args()


def denormalize(tensor):
    mean = torch.tensor([0.485, 0.456, 0.406], device=tensor.device).view(3,1,1)
    std = torch.tensor([0.229, 0.224, 0.225], device=tensor.device).view(3,1,1)
    x = (tensor * std + mean).clamp(0, 1)
    return x.permute(1,2,0).detach().cpu().numpy()


def gradcam(model, tensor, target):
    holder = {}
    def hook(_module, _inputs, output):
        holder["activation"] = output
        output.register_hook(lambda grad: holder.__setitem__("gradient", grad))
    handle = model.features[-1].register_forward_hook(hook)
    try:
        model.zero_grad(set_to_none=True)
        logits = model(tensor.unsqueeze(0))
        logits[0, int(target)].backward()
        act = holder["activation"]
        grad = holder["gradient"]
        weights = grad.mean(dim=(2,3), keepdim=True)
        cam = torch.relu((weights * act).sum(1, keepdim=True))
        cam = torch.nn.functional.interpolate(
            cam, size=tensor.shape[-2:], mode="bilinear", align_corners=False
        )[0,0]
        cam = cam - cam.min()
        cam_max = cam.detach().max().item()
        if cam_max > 0:
            cam = cam / cam_max
        return cam.detach().cpu().numpy()
    finally:
        handle.remove()


def overlay(rgb, cam):
    heat = plt.get_cmap("jet")(cam)[...,:3]
    return np.clip(0.58*rgb + 0.42*heat, 0, 1)


def make_sheet(model, frame, device, path, count):
    frame = frame.head(count).copy()
    if frame.empty:
        return
    size, label_h, cols = 224, 74, 2
    card_w, card_h = size*2, size+label_h
    rows = math.ceil(len(frame)/cols)
    canvas = Image.new("RGB", (cols*card_w, rows*card_h), "white")
    draw = ImageDraw.Draw(canvas)

    for k, (_, row) in enumerate(tqdm(frame.iterrows(), total=len(frame), desc="Grad-CAM", unit="img")):
        with Image.open(DATASET_ROOT / str(row["relative_path"])) as im:
            tensor = evaluation_transform(im.convert("RGB")).to(device)
        rgb = denormalize(tensor)
        cam = gradcam(model, tensor, int(row["predicted_index"]))
        ov = overlay(rgb, cam)
        original = Image.fromarray((rgb*255).astype(np.uint8))
        heat = Image.fromarray((ov*255).astype(np.uint8))
        x0, y0 = (k % cols)*card_w, (k // cols)*card_h
        canvas.paste(original, (x0, y0))
        canvas.paste(heat, (x0+size, y0))
        label = (
            f"conf={row['confidence']:.3f} | {'OK' if bool(row['correct']) else 'ERROR'}\n"
            f"T:{str(row['true_class'])[:38]}\nP:{str(row['predicted_class'])[:38]}"
        )
        draw.multiline_text((x0+3, y0+size+3), label, fill="black", spacing=2)
    canvas.save(path, quality=93)


def main():
    args = parse_args()
    if args.images_per_sheet <= 0:
        raise ValueError("--images-per-sheet must be positive.")
    pred_path = OUT / "efficientnet_predictions.csv"
    if not pred_path.is_file():
        raise FileNotFoundError("Run src/full_control_core.py first.")
    frame = pd.read_csv(pred_path)
    checkpoint = torch.load(EFF_CKPT, map_location="cpu", weights_only=False)
    class_names = list(checkpoint["class_names"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_efficientnet_b0_transfer(
        len(class_names), float(checkpoint.get("dropout_rate", 0.3)), False
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    correct = frame[frame["correct"].astype(bool)].sort_values("confidence", ascending=False)
    errors = frame[~frame["correct"].astype(bool)].sort_values("confidence", ascending=False)
    correct.head(args.images_per_sheet).to_csv(OUT / "gradcam_selected_correct.csv", index=False)
    errors.head(args.images_per_sheet).to_csv(OUT / "gradcam_selected_errors.csv", index=False)
    make_sheet(model, correct, device, FIG / "efficientnet_gradcam_correct.jpg", args.images_per_sheet)
    make_sheet(model, errors, device, FIG / "efficientnet_gradcam_errors.jpg", args.images_per_sheet)

    print("="*80)
    print("GRAD-CAM VISUAL AUDIT COMPLETE")
    print("="*80)
    print("Correct:", FIG / "efficientnet_gradcam_correct.jpg")
    print("Errors:", FIG / "efficientnet_gradcam_errors.jpg")
    print("Interpretation: attention on leaf/disease regions is supportive evidence only;")
    print("Grad-CAM alone cannot prove absence of shortcut learning.")


if __name__ == "__main__":
    main()
