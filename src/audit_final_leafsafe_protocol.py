from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "PlantVillage"
MANIFEST_PATH = PROJECT_ROOT / "outputs" / "audit" / "official_leaf_safe_split_manifest.csv"
SUMMARY_PATH = PROJECT_ROOT / "outputs" / "audit" / "official_leaf_safe_split_summary.json"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "audit" / "final_leafsafe_protocol_audit"

STRICT_DHASH_THRESHOLD = 4
REVIEW_DHASH_THRESHOLD = 8
VERY_HIGH_EMBEDDING_SIMILARITY = 0.9995
HIGH_EMBEDDING_SIMILARITY = 0.9950


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Final audit of the conservative official PlantVillage manifest: "
            "exact overlap, mapped physical-leaf overlap, dHash near-duplicates, "
            "and ImageNet feature-neighbor review."
        )
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--top-embedding-pairs", type=int, default=200)
    parser.add_argument("--contact-sheet-pairs", type=int, default=24)
    return parser.parse_args()


def validate_args(args):
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative.")
    if args.top_embedding_pairs <= 0:
        raise ValueError("--top-embedding-pairs must be positive.")
    if args.contact_sheet_pairs < 0:
        raise ValueError("--contact-sheet-pairs cannot be negative.")


def load_manifest_and_summary():
    if not MANIFEST_PATH.is_file() or not SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            "Final leaf-safe manifest/summary missing. Run "
            "src/build_official_leaf_safe_manifest.py first."
        )

    frame = pd.read_csv(MANIFEST_PATH, keep_default_na=False)
    with SUMMARY_PATH.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)

    required = {
        "split", "relative_path", "class_index", "class_name", "leaf_id", "sha256"
    }
    missing = required - set(frame.columns)
    if missing:
        raise RuntimeError("Manifest missing columns: " + ", ".join(sorted(missing)))

    if set(frame["split"].unique()) != {"train", "validation", "test"}:
        raise RuntimeError("Manifest split names are invalid.")
    if int(frame["relative_path"].duplicated().sum()) != 0:
        raise RuntimeError("Manifest contains duplicated local paths.")

    missing_files = [
        p for p in frame["relative_path"] if not (DATASET_ROOT / str(p)).is_file()
    ]
    if missing_files:
        raise FileNotFoundError(f"Manifest references {len(missing_files)} missing files.")

    return frame.reset_index(drop=True), summary


def split_pair(a: str, b: str) -> str:
    return "--".join(sorted((a, b)))


def exact_hash_audit(frame):
    groups = defaultdict(list)
    for index, row in frame.iterrows():
        groups[str(row["sha256"])].append(index)

    rows = []
    for digest, indices in groups.items():
        if len({frame.iloc[i]["split"] for i in indices}) <= 1:
            continue
        for left_pos in range(len(indices)):
            for right_pos in range(left_pos + 1, len(indices)):
                left = frame.iloc[indices[left_pos]]
                right = frame.iloc[indices[right_pos]]
                if left["split"] == right["split"]:
                    continue
                rows.append({
                    "sha256": digest,
                    "split_pair": split_pair(str(left["split"]), str(right["split"])),
                    "class_a": left["class_name"],
                    "class_b": right["class_name"],
                    "path_a": left["relative_path"],
                    "path_b": right["relative_path"],
                })
    return pd.DataFrame(rows)


def mapped_leaf_audit(frame):
    mapped = frame[frame["leaf_id"].astype(str) != ""].copy()
    rows = []
    for leaf_id, group in mapped.groupby("leaf_id"):
        splits = sorted(group["split"].unique())
        if len(splits) <= 1:
            continue
        rows.append({
            "leaf_id": leaf_id,
            "splits": "|".join(splits),
            "image_count": len(group),
            "classes": "|".join(sorted(group["class_name"].unique())),
            "paths": "|".join(group["relative_path"].astype(str)),
        })
    return pd.DataFrame(rows)


def dhash64(path: Path) -> int:
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(image, dtype=np.int16)
    differences = pixels[:, :-1] > pixels[:, 1:]
    value = 0
    for bit, flag in enumerate(differences.reshape(-1)):
        if bool(flag):
            value |= 1 << bit
    return value


def hamming64(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def bands(value: int, count: int):
    base = 64 // count
    remainder = 64 % count
    shift = 0
    result = []
    for band_index in range(count):
        size = base + (1 if band_index < remainder else 0)
        mask = (1 << size) - 1
        result.append((band_index, size, (value >> shift) & mask))
        shift += size
    return result


def perceptual_audit(frame):
    hashes = [0] * len(frame)
    for index, row in tqdm(
        frame.iterrows(), total=len(frame), desc="dHash calculation", unit="img"
    ):
        hashes[index] = dhash64(DATASET_ROOT / str(row["relative_path"]))

    # With threshold T and T+1 bands, every Hamming-distance <= T pair shares
    # at least one exact band. Same-class bucketing reduces false candidates;
    # global byte-identical overlap is already checked separately via SHA-256.
    bucket_count = REVIEW_DHASH_THRESHOLD + 1
    buckets_by_key = defaultdict(list)
    seen = set()
    rows = []

    for index, row in tqdm(
        frame.iterrows(), total=len(frame), desc="Cross-split dHash matching", unit="img"
    ):
        value = hashes[index]
        class_index = int(row["class_index"])
        candidates = set()
        for band_index, size, band_value in bands(value, bucket_count):
            candidates.update(
                buckets_by_key[(class_index, band_index, size, band_value)]
            )

        for other_index in candidates:
            other = frame.iloc[other_index]
            if other["split"] == row["split"]:
                continue
            pair_key = (min(index, other_index), max(index, other_index))
            if pair_key in seen:
                continue
            seen.add(pair_key)
            distance = hamming64(value, hashes[other_index])
            if distance > REVIEW_DHASH_THRESHOLD:
                continue
            leaf_a = str(other["leaf_id"])
            leaf_b = str(row["leaf_id"])
            rows.append({
                "hamming_distance": distance,
                "strict_threshold_match": distance <= STRICT_DHASH_THRESHOLD,
                "split_pair": split_pair(str(other["split"]), str(row["split"])),
                "class_name": row["class_name"],
                "path_a": other["relative_path"],
                "path_b": row["relative_path"],
                "leaf_id_a": leaf_a,
                "leaf_id_b": leaf_b,
                "involves_unmapped_leaf": leaf_a == "" or leaf_b == "",
            })

        for band_index, size, band_value in bands(value, bucket_count):
            buckets_by_key[(class_index, band_index, size, band_value)].append(index)

    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(
            ["hamming_distance", "split_pair", "class_name", "path_a", "path_b"]
        ).reset_index(drop=True)
    return result


class EmbeddingDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = DATASET_ROOT / str(row["relative_path"])
        with Image.open(path) as image:
            image = self.transform(image.convert("RGB"))
        return image, index


def extract_embeddings(frame, batch_size, num_workers):
    weights = ResNet18_Weights.DEFAULT
    dataset = EmbeddingDataset(frame, weights.transforms())
    options = {
        "batch_size": batch_size,
        "shuffle": False,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    if num_workers > 0:
        options.update({"persistent_workers": True, "prefetch_factor": 2})
    loader = DataLoader(dataset, **options)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    backbone = resnet18(weights=weights)
    feature_model = nn.Sequential(*list(backbone.children())[:-1]).to(device)
    feature_model.eval()
    embeddings = np.empty((len(frame), 512), dtype=np.float32)

    with torch.inference_mode():
        for images, indices in tqdm(loader, desc="ImageNet ResNet18 embeddings", unit="batch"):
            images = images.to(device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=device.type == "cuda",
            ):
                features = feature_model(images)
            features = torch.nn.functional.normalize(features.flatten(1).float(), p=2, dim=1)
            embeddings[indices.numpy()] = features.cpu().numpy()

    return embeddings, str(device)


def embedding_neighbor_audit(frame, embeddings, top_n):
    rows = []
    split_pairs = [("train", "validation"), ("train", "test"), ("validation", "test")]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for class_index in sorted(frame["class_index"].astype(int).unique()):
        class_frame = frame[frame["class_index"].astype(int) == class_index]
        class_name = str(class_frame.iloc[0]["class_name"])
        for split_a, split_b in split_pairs:
            idx_a = class_frame.index[class_frame["split"] == split_a].to_numpy()
            idx_b = class_frame.index[class_frame["split"] == split_b].to_numpy()
            if len(idx_a) == 0 or len(idx_b) == 0:
                continue
            if len(idx_a) <= len(idx_b):
                query_indices, candidate_indices = idx_a, idx_b
            else:
                query_indices, candidate_indices = idx_b, idx_a

            candidate_tensor = torch.from_numpy(embeddings[candidate_indices]).to(device)
            for start in range(0, len(query_indices), 512):
                block_indices = query_indices[start : start + 512]
                query_tensor = torch.from_numpy(embeddings[block_indices]).to(device)
                similarity = query_tensor @ candidate_tensor.T
                values, positions = similarity.max(dim=1)
                values = values.cpu().numpy()
                positions = positions.cpu().numpy()

                for pos, query_index in enumerate(block_indices):
                    candidate_index = candidate_indices[int(positions[pos])]
                    left = frame.iloc[int(query_index)]
                    right = frame.iloc[int(candidate_index)]
                    score = float(values[pos])
                    leaf_a = str(left["leaf_id"])
                    leaf_b = str(right["leaf_id"])
                    rows.append({
                        "cosine_similarity": score,
                        "very_high_similarity": score >= VERY_HIGH_EMBEDDING_SIMILARITY,
                        "high_similarity": score >= HIGH_EMBEDDING_SIMILARITY,
                        "split_pair": split_pair(str(left["split"]), str(right["split"])),
                        "class_name": class_name,
                        "path_a": left["relative_path"],
                        "path_b": right["relative_path"],
                        "leaf_id_a": leaf_a,
                        "leaf_id_b": leaf_b,
                        "involves_unmapped_leaf": leaf_a == "" or leaf_b == "",
                    })
            del candidate_tensor
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["pair_key"] = result.apply(
        lambda row: "||".join(sorted([str(row["path_a"]), str(row["path_b"])])),
        axis=1,
    )
    result = (
        result.sort_values("cosine_similarity", ascending=False)
        .drop_duplicates("pair_key")
        .drop(columns="pair_key")
        .reset_index(drop=True)
    )
    return result.head(top_n).copy()


def contact_sheet(pairs, output_path, score_column, max_pairs):
    if pairs.empty or max_pairs <= 0:
        return
    pairs = pairs.head(max_pairs)
    image_size = 180
    label_height = 70
    card_width = image_size * 2
    card_height = image_size + label_height
    columns = 2
    row_count = int(np.ceil(len(pairs) / columns))
    canvas = Image.new("RGB", (card_width * columns, card_height * row_count), "white")
    draw = ImageDraw.Draw(canvas)

    for card_index, (_, row) in enumerate(pairs.iterrows()):
        x0 = (card_index % columns) * card_width
        y0 = (card_index // columns) * card_height
        for offset, key in enumerate(("path_a", "path_b")):
            with Image.open(DATASET_ROOT / str(row[key])) as image:
                image = image.convert("RGB")
                image.thumbnail((image_size, image_size))
                slot = Image.new("RGB", (image_size, image_size), "white")
                slot.paste(image, ((image_size - image.width) // 2, (image_size - image.height) // 2))
                canvas.paste(slot, (x0 + offset * image_size, y0))
        label = (
            f"{card_index + 1}. {score_column}={row[score_column]}\n"
            f"{row['split_pair']} | {row['class_name']}\n"
            f"unmapped_leaf={bool(row.get('involves_unmapped_leaf', False))}"
        )
        draw.text((x0 + 4, y0 + image_size + 4), label, fill="black")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, quality=92)


def main():
    args = parse_args()
    validate_args(args)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame, source_summary = load_manifest_and_summary()
    counts = frame["split"].value_counts().to_dict()
    mapped = int((frame["leaf_id"].astype(str) != "").sum())
    unmapped = len(frame) - mapped

    print("=" * 80)
    print("FINAL CONSERVATIVE OFFICIAL LEAF-SAFE AUDIT")
    print("=" * 80)
    print("Train:", counts.get("train", 0))
    print("Validation:", counts.get("validation", 0))
    print("Locked official test:", counts.get("test", 0))
    print(f"Mapped leaf metadata: {mapped}/{len(frame)} ({mapped / len(frame) * 100:.2f}%)")
    print("Unmapped leaf metadata:", unmapped)
    print()

    exact = exact_hash_audit(frame)
    exact.to_csv(OUTPUT_DIR / "exact_cross_split_pairs.csv", index=False)

    leaf = mapped_leaf_audit(frame)
    leaf.to_csv(OUTPUT_DIR / "mapped_leaf_cross_split_groups.csv", index=False)

    dhash = perceptual_audit(frame)
    dhash.to_csv(OUTPUT_DIR / "dhash_cross_split_review_pairs.csv", index=False)
    if dhash.empty:
        strict = dhash.copy()
    else:
        strict = dhash[dhash["strict_threshold_match"]].copy()
    strict.to_csv(OUTPUT_DIR / "dhash_strict_cross_split_pairs.csv", index=False)
    contact_sheet(
        dhash,
        OUTPUT_DIR / "dhash_review_contact_sheet.jpg",
        "hamming_distance",
        args.contact_sheet_pairs,
    )

    embedding_pairs = None
    embedding_device = None
    if not args.skip_embedding:
        embeddings, embedding_device = extract_embeddings(
            frame, args.batch_size, args.num_workers
        )
        embedding_pairs = embedding_neighbor_audit(
            frame, embeddings, args.top_embedding_pairs
        )
        embedding_pairs.to_csv(
            OUTPUT_DIR / "embedding_top_cross_split_neighbors.csv", index=False
        )
        contact_sheet(
            embedding_pairs,
            OUTPUT_DIR / "embedding_top_pairs_contact_sheet.jpg",
            "cosine_similarity",
            args.contact_sheet_pairs,
        )

    exact_pass = len(exact) == 0
    leaf_pass = len(leaf) == 0
    dhash_pass = len(strict) == 0
    unmapped_strict = 0 if strict.empty else int(strict["involves_unmapped_leaf"].sum())

    if embedding_pairs is None or embedding_pairs.empty:
        embedding_max = None
        very_high = None
        high = None
        unmapped_very_high = None
    else:
        embedding_max = float(embedding_pairs["cosine_similarity"].max())
        very_high = int(embedding_pairs["very_high_similarity"].sum())
        high = int(embedding_pairs["high_similarity"].sum())
        unmapped_very_high = int(
            (embedding_pairs["very_high_similarity"] & embedding_pairs["involves_unmapped_leaf"]).sum()
        )

    summary = {
        "protocol": source_summary.get("protocol"),
        "manifest_images": len(frame),
        "split_counts": {k: int(v) for k, v in counts.items()},
        "mapped_leaf_metadata_images": mapped,
        "unmapped_leaf_metadata_images": unmapped,
        "mapped_leaf_metadata_coverage": mapped / len(frame),
        "exact_cross_split_pairs": len(exact),
        "mapped_leaf_cross_split_groups": len(leaf),
        "strict_dhash_threshold": STRICT_DHASH_THRESHOLD,
        "review_dhash_threshold": REVIEW_DHASH_THRESHOLD,
        "strict_dhash_cross_split_pairs": len(strict),
        "strict_dhash_pairs_involving_unmapped_leaf": unmapped_strict,
        "review_dhash_cross_split_pairs": len(dhash),
        "embedding_audit_skipped": args.skip_embedding,
        "embedding_device": embedding_device,
        "embedding_max_cosine_similarity": embedding_max,
        "embedding_very_high_threshold": VERY_HIGH_EMBEDDING_SIMILARITY,
        "embedding_high_threshold": HIGH_EMBEDDING_SIMILARITY,
        "embedding_very_high_pairs_in_top_saved": very_high,
        "embedding_high_pairs_in_top_saved": high,
        "embedding_unmapped_very_high_pairs_in_top_saved": unmapped_very_high,
        "hard_checks": {
            "exact_duplicate_pass": exact_pass,
            "mapped_leaf_overlap_pass": leaf_pass,
            "strict_perceptual_near_duplicate_pass": dhash_pass,
        },
        "hard_audit_pass": bool(exact_pass and leaf_pass and dhash_pass),
        "notes": {
            "embedding": (
                "Embedding similarity is heuristic and must be visually reviewed; "
                "it is not proof of physical-leaf identity."
            ),
            "leaf_metadata": (
                "Leaf-map coverage is incomplete. The official PlantVillage test "
                "split is documented as leaf-preserving; this audit independently "
                "checks mapped leaf IDs and also screens unmapped samples via exact, "
                "perceptual, and feature similarity."
            ),
        },
    }

    summary_path = OUTPUT_DIR / "final_audit_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print("=" * 80)
    print("FINAL AUDIT SUMMARY")
    print("=" * 80)
    print("Exact cross-split duplicate pairs:", len(exact))
    print("Mapped physical-leaf cross-split groups:", len(leaf))
    print(f"Strict dHash near-duplicate pairs (<= {STRICT_DHASH_THRESHOLD}):", len(strict))
    print("  involving unmapped leaf metadata:", unmapped_strict)
    print(f"dHash review pool (<= {REVIEW_DHASH_THRESHOLD}):", len(dhash))
    if embedding_pairs is None:
        print("Embedding audit: SKIPPED")
    elif embedding_pairs.empty:
        print("Embedding audit: no pairs produced")
    else:
        print("Embedding max cosine similarity:", f"{embedding_max:.6f}")
        print(f"Embedding pairs >= {VERY_HIGH_EMBEDDING_SIMILARITY:.4f} (top saved):", very_high)
        print(f"Embedding pairs >= {HIGH_EMBEDDING_SIMILARITY:.4f} (top saved):", high)
        print("Very-high pairs involving unmapped leaf metadata:", unmapped_very_high)

    print()
    print("Exact duplicate check:", "PASS" if exact_pass else "FAIL")
    print("Mapped leaf overlap check:", "PASS" if leaf_pass else "FAIL")
    print("Strict perceptual near-duplicate check:", "PASS" if dhash_pass else "REVIEW/FAIL")
    print("HARD AUDIT:", "PASS" if summary["hard_audit_pass"] else "NOT PASSED")
    print()
    print("dHash review sheet:", OUTPUT_DIR / "dhash_review_contact_sheet.jpg")
    print("Embedding review sheet:", OUTPUT_DIR / "embedding_top_pairs_contact_sheet.jpg")
    print("Summary:", summary_path)


if __name__ == "__main__":
    main()
