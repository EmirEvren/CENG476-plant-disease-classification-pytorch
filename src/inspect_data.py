from collections import Counter
from pathlib import Path

import pandas as pd
from torchvision.datasets import ImageFolder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "PlantVillage"

TRAIN_DIR = DATASET_ROOT / "train"
VAL_DIR = DATASET_ROOT / "val"

OUTPUT_FILE = PROJECT_ROOT / "outputs" / "class_distribution.csv"


if not TRAIN_DIR.exists():
    raise FileNotFoundError(f"Train klasörü bulunamadı: {TRAIN_DIR}")

if not VAL_DIR.exists():
    raise FileNotFoundError(f"Val klasörü bulunamadı: {VAL_DIR}")


train_dataset = ImageFolder(TRAIN_DIR)
val_dataset = ImageFolder(VAL_DIR)

if train_dataset.class_to_idx != val_dataset.class_to_idx:
    raise ValueError("Train ve val sınıfları aynı değil.")


train_counts = Counter(train_dataset.targets)
val_counts = Counter(val_dataset.targets)

rows = []

for class_name, class_index in train_dataset.class_to_idx.items():
    train_count = train_counts[class_index]
    val_count = val_counts[class_index]

    rows.append(
        {
            "class_name": class_name,
            "train_count": train_count,
            "val_count": val_count,
            "total_count": train_count + val_count,
        }
    )

distribution = pd.DataFrame(rows)
distribution = distribution.sort_values("class_name")

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
distribution.to_csv(OUTPUT_FILE, index=False)

print("=" * 55)
print("PLANTVILLAGE VERİ SETİ ÖZETİ")
print("=" * 55)
print("Sınıf sayısı:", len(train_dataset.classes))
print("Train görüntü sayısı:", len(train_dataset))
print("Val görüntü sayısı:", len(val_dataset))
print("Toplam görüntü sayısı:", len(train_dataset) + len(val_dataset))
print("=" * 55)
print(distribution.to_string(index=False))
print("=" * 55)
print("Sınıf dağılımı kaydedildi:", OUTPUT_FILE)