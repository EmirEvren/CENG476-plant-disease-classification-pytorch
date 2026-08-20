from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ULTRA_MANIFEST = PROJECT_ROOT / "outputs" / "audit" / "official_leaf_safe_split_manifest_ultrastrict.csv"
ULTRA_SUMMARY = PROJECT_ROOT / "outputs" / "audit" / "official_leaf_safe_split_summary_ultrastrict.json"


def parse_args():
    p = argparse.ArgumentParser(
        description="Run EfficientNet-B0 on the fixed ultra-strict manifest with a chosen random seed."
    )
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--run-name", type=str, required=True)
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.num_workers < 0:
        raise ValueError("Invalid training arguments.")
    if not ULTRA_MANIFEST.is_file() or not ULTRA_SUMMARY.is_file():
        raise FileNotFoundError("Ultra-strict manifest/summary missing.")

    import official_data_setup
    official_data_setup.MANIFEST_PATH = ULTRA_MANIFEST
    official_data_setup.SUMMARY_PATH = ULTRA_SUMMARY
    official_data_setup.SEED = args.seed

    import train_efficientnet_official as trainer
    trainer.SEED = args.seed

    sys.argv = [
        "train_efficientnet_official",
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--backbone-learning-rate", "1e-4",
        "--classifier-learning-rate", "5e-4",
        "--weight-decay", "1e-4",
        "--dropout", "0.3",
        "--run-name", args.run_name,
    ]
    print("="*80)
    print("ULTRA-STRICT EFFICIENTNET SEED RUN")
    print("="*80)
    print("Seed:", args.seed)
    print("Run:", args.run_name)
    print("Manifest is fixed; only stochastic training state changes.")
    trainer.main()


if __name__ == "__main__":
    main()
