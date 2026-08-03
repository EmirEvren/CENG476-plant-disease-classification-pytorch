from pathlib import Path

import kagglehub


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

print("PlantVillage veri seti indiriliyor...")

dataset_path = kagglehub.dataset_download(
    "mohitsingh1804/plantvillage",
    output_dir=str(RAW_DATA_DIR),
)

print("İndirme tamamlandı.")
print("Veri seti konumu:", dataset_path)