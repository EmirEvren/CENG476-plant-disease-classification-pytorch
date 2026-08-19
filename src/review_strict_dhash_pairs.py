from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "PlantVillage"
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit" / "final_leafsafe_protocol_audit"
STRICT_PATH = AUDIT_DIR / "dhash_strict_cross_split_pairs.csv"
OUTPUT_CSV = AUDIT_DIR / "strict_dhash_pair_detailed_review.csv"
OUTPUT_JSON = AUDIT_DIR / "strict_dhash_pair_review_summary.json"
OUTPUT_SHEET = AUDIT_DIR / "strict_dhash_unmapped_contact_sheet.jpg"

# These are review flags only, not formal leakage definitions.
PIXEL_CORRELATION_REVIEW = 0.995
RGB_MAE_REVIEW = 0.03
EMBEDDING_REVIEW = 0.990


def load_pairs():
    if not STRICT_PATH.is_file():
        raise FileNotFoundError(
            f"Strict dHash pair CSV not found: {STRICT_PATH}\n"
            "Run src/audit_final_leafsafe_protocol.py first."
        )
    frame = pd.read_csv(STRICT_PATH, keep_default_na=False)
    required = {
        "hamming_distance",
        "split_pair",
        "class_name",
        "path_a",
        "path_b",
        "leaf_id_a",
        "leaf_id_b",
        "involves_unmapped_leaf",
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError("Strict-pair CSV missing columns: " + ", ".join(sorted(missing)))
    return frame


def load_standardized_rgb(relative_path: str, size=256):
    with Image.open(DATASET_ROOT / relative_path) as image:
        image = image.convert("RGB").resize((size, size), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0


def pixel_metrics(path_a: str, path_b: str):
    a = load_standardized_rgb(path_a)
    b = load_standardized_rgb(path_b)
    mae = float(np.abs(a - b).mean())

    gray_a = a.mean(axis=2).reshape(-1)
    gray_b = b.mean(axis=2).reshape(-1)
    std_a = float(gray_a.std())
    std_b = float(gray_b.std())
    if std_a == 0.0 or std_b == 0.0:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(gray_a, gray_b)[0, 1])
    return mae, correlation


def build_feature_model(device):
    weights = ResNet18_Weights.DEFAULT
    backbone = resnet18(weights=weights)
    model = nn.Sequential(*list(backbone.children())[:-1]).to(device)
    model.eval()
    return model, weights.transforms()


def embed_unique_paths(paths):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = build_feature_model(device)
    result = {}

    with torch.inference_mode():
        for relative_path in tqdm(paths, desc="Embedding strict-pair images", unit="img"):
            with Image.open(DATASET_ROOT / relative_path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                feature = model(tensor).flatten(1).float()
            feature = torch.nn.functional.normalize(feature, p=2, dim=1)
            result[relative_path] = feature.squeeze(0).cpu().numpy()

    return result, str(device)


def cosine(a, b):
    return float(np.dot(a, b))


def make_contact_sheet(frame, output_path):
    if frame.empty:
        return

    image_size = 220
    label_height = 105
    card_width = image_size * 2
    card_height = image_size + label_height
    columns = 2
    rows = int(np.ceil(len(frame) / columns))
    canvas = Image.new("RGB", (card_width * columns, card_height * rows), "white")
    draw = ImageDraw.Draw(canvas)

    for card_index, (_, row) in enumerate(frame.iterrows()):
        x0 = (card_index % columns) * card_width
        y0 = (card_index // columns) * card_height

        for offset, key in enumerate(("path_a", "path_b")):
            with Image.open(DATASET_ROOT / str(row[key])) as image:
                image = image.convert("RGB")
                image.thumbnail((image_size, image_size))
                slot = Image.new("RGB", (image_size, image_size), "white")
                slot.paste(
                    image,
                    ((image_size - image.width) // 2, (image_size - image.height) // 2),
                )
                canvas.paste(slot, (x0 + offset * image_size, y0))

        label = (
            f"#{card_index + 1} dHash={int(row['hamming_distance'])} | {row['split_pair']}\n"
            f"embed={row['embedding_cosine']:.4f} | corr={row['gray_pixel_correlation']:.4f} | "
            f"MAE={row['rgb_mae']:.4f}\n"
            f"{row['class_name']}\n"
            f"leafA={row['leaf_id_a'] or 'UNMAPPED'} | leafB={row['leaf_id_b'] or 'UNMAPPED'}"
        )
        draw.text((x0 + 4, y0 + image_size + 4), label, fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=94)


def main():
    pairs = load_pairs()

    print("=" * 80)
    print("FOCUSED REVIEW OF STRICT dHASH CROSS-SPLIT PAIRS")
    print("=" * 80)
    print("Strict pairs:", len(pairs))
    print("Split-pair distribution:", dict(Counter(pairs["split_pair"])))
    print(
        "Pairs involving unmapped leaf metadata:",
        int(pairs["involves_unmapped_leaf"].astype(bool).sum()),
    )

    mapped_pairs = pairs[~pairs["involves_unmapped_leaf"].astype(bool)].copy()
    if not mapped_pairs.empty:
        same_mapped_leaf = int(
            (
                mapped_pairs["leaf_id_a"].astype(str)
                == mapped_pairs["leaf_id_b"].astype(str)
            ).sum()
        )
    else:
        same_mapped_leaf = 0
    print("Mapped strict pairs with same leaf ID:", same_mapped_leaf)

    unique_paths = sorted(set(pairs["path_a"]) | set(pairs["path_b"]))
    embeddings, device = embed_unique_paths(unique_paths)

    detailed_rows = []
    for _, row in tqdm(pairs.iterrows(), total=len(pairs), desc="Pair-level metrics", unit="pair"):
        path_a = str(row["path_a"])
        path_b = str(row["path_b"])
        mae, correlation = pixel_metrics(path_a, path_b)
        embedding_similarity = cosine(embeddings[path_a], embeddings[path_b])

        pixel_review_flag = (
            correlation >= PIXEL_CORRELATION_REVIEW
            and mae <= RGB_MAE_REVIEW
        )
        embedding_review_flag = embedding_similarity >= EMBEDDING_REVIEW

        detailed_rows.append(
            {
                **row.to_dict(),
                "embedding_cosine": embedding_similarity,
                "gray_pixel_correlation": correlation,
                "rgb_mae": mae,
                "pixel_review_flag": bool(pixel_review_flag),
                "embedding_review_flag": bool(embedding_review_flag),
                "manual_review_priority": bool(
                    row["involves_unmapped_leaf"]
                    or pixel_review_flag
                    or embedding_review_flag
                ),
            }
        )

    detailed = pd.DataFrame(detailed_rows).sort_values(
        [
            "involves_unmapped_leaf",
            "embedding_cosine",
            "gray_pixel_correlation",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    detailed.to_csv(OUTPUT_CSV, index=False)

    unmapped = detailed[detailed["involves_unmapped_leaf"].astype(bool)].copy()
    make_contact_sheet(unmapped, OUTPUT_SHEET)

    summary = {
        "strict_pairs": int(len(detailed)),
        "strict_pair_split_distribution": {
            str(key): int(value)
            for key, value in Counter(detailed["split_pair"]).items()
        },
        "pairs_involving_unmapped_leaf": int(len(unmapped)),
        "fully_mapped_pairs": int(len(detailed) - len(unmapped)),
        "fully_mapped_pairs_same_leaf_id": same_mapped_leaf,
        "max_embedding_cosine_all_strict": (
            None if detailed.empty else float(detailed["embedding_cosine"].max())
        ),
        "max_embedding_cosine_unmapped": (
            None if unmapped.empty else float(unmapped["embedding_cosine"].max())
        ),
        "max_gray_pixel_correlation_unmapped": (
            None if unmapped.empty else float(unmapped["gray_pixel_correlation"].max())
        ),
        "min_rgb_mae_unmapped": (
            None if unmapped.empty else float(unmapped["rgb_mae"].min())
        ),
        "embedding_review_threshold": EMBEDDING_REVIEW,
        "pixel_correlation_review_threshold": PIXEL_CORRELATION_REVIEW,
        "rgb_mae_review_threshold": RGB_MAE_REVIEW,
        "unmapped_embedding_review_flags": int(
            unmapped["embedding_review_flag"].sum()
        ),
        "unmapped_pixel_review_flags": int(unmapped["pixel_review_flag"].sum()),
        "device": device,
        "interpretation": (
            "dHash is a coarse perceptual screen. Fully mapped pairs with different "
            "leaf IDs are not same-leaf leakage according to available metadata. "
            "Pairs involving unmapped leaf metadata require visual review; embedding "
            "and pixel metrics are supporting heuristics, not proof of identity."
        ),
    }

    with OUTPUT_JSON.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("=" * 80)
    print("STRICT PAIR REVIEW SUMMARY")
    print("=" * 80)
    print("Fully mapped strict pairs:", summary["fully_mapped_pairs"])
    print("  same mapped leaf ID:", same_mapped_leaf)
    print("Unmapped-involved strict pairs:", len(unmapped))
    if not detailed.empty:
        print("Max embedding cosine, all strict:", f"{summary['max_embedding_cosine_all_strict']:.6f}")
    if not unmapped.empty:
        print("Max embedding cosine, unmapped:", f"{summary['max_embedding_cosine_unmapped']:.6f}")
        print("Max grayscale correlation, unmapped:", f"{summary['max_gray_pixel_correlation_unmapped']:.6f}")
        print("Min RGB MAE, unmapped:", f"{summary['min_rgb_mae_unmapped']:.6f}")
        print("Unmapped embedding review flags:", summary["unmapped_embedding_review_flags"])
        print("Unmapped pixel review flags:", summary["unmapped_pixel_review_flags"])
    print("Detailed CSV:", OUTPUT_CSV)
    print("Unmapped contact sheet:", OUTPUT_SHEET)
    print("Summary JSON:", OUTPUT_JSON)
    print()
    print(
        "NEXT: visually inspect/upload strict_dhash_unmapped_contact_sheet.jpg. "
        "Only the unmapped-involved pairs remain unresolved by leaf metadata."
    )


if __name__ == "__main__":
    main()
