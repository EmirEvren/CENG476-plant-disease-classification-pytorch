from __future__ import annotations

import importlib
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = PROJECT_ROOT / "outputs" / "audit"
ULTRA_MANIFEST = AUDIT_DIR / "official_leaf_safe_split_manifest_ultrastrict.csv"
ULTRA_SUMMARY = AUDIT_DIR / "official_leaf_safe_split_summary_ultrastrict.json"

ALLOWED_MODULES = {
    "official_data_setup",
    "train_baseline_official",
    "evaluate_baseline_official",
    "train_resnet18_official",
    "evaluate_resnet18_official",
    "train_efficientnet_official",
    "evaluate_efficientnet_official",
    "evaluate_ensemble_official",
}


def activate_ultrastrict_manifest():
    if not ULTRA_MANIFEST.is_file() or not ULTRA_SUMMARY.is_file():
        raise FileNotFoundError(
            "Ultra-strict manifest/summary missing. Run "
            "src/build_ultrastrict_manifest.py first."
        )

    import official_data_setup

    official_data_setup.MANIFEST_PATH = ULTRA_MANIFEST
    official_data_setup.SUMMARY_PATH = ULTRA_SUMMARY
    return official_data_setup


def checkpoint_run_name(arguments):
    if "--checkpoint" not in arguments:
        return None
    index = arguments.index("--checkpoint")
    if index + 1 >= len(arguments):
        return None
    stem = Path(arguments[index + 1]).stem
    return stem[:-5] if stem.endswith("_best") else stem


def main():
    if len(sys.argv) < 2:
        raise SystemExit(
            "Usage: python src/ultrastrict_entrypoint.py <module> [module args...]"
        )

    module_name = sys.argv[1]
    if module_name not in ALLOWED_MODULES:
        raise ValueError(
            f"Unsupported module: {module_name}. Allowed: {sorted(ALLOWED_MODULES)}"
        )

    remaining_args = sys.argv[2:]
    activate_ultrastrict_manifest()

    # Present the remaining CLI arguments to the underlying script unchanged.
    sys.argv = [module_name, *remaining_args]
    module = importlib.import_module(module_name)

    # EfficientNet's original evaluator historically used a fixed output run
    # name. Override it here so ultra-strict results never overwrite the prior
    # leaf-safe evaluation directory.
    if module_name == "evaluate_efficientnet_official":
        run_name = checkpoint_run_name(remaining_args)
        if run_name:
            module.RUN_NAME = run_name

    # The ensemble evaluator also historically used a fixed output run name.
    if module_name == "evaluate_ensemble_official":
        module.RUN_NAME = "resnet18_efficientnet_official_ultrastrict_soft_voting"

    if not hasattr(module, "main"):
        raise RuntimeError(f"Module has no main(): {module_name}")

    module.main()


if __name__ == "__main__":
    main()
