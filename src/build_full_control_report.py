from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT = PROJECT_ROOT / "outputs" / "audit" / "full_control"
ULTRA = PROJECT_ROOT / "outputs" / "audit" / "official_leaf_safe_split_summary_ultrastrict.json"


def load_json(path):
    if not path.is_file():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def pct(x):
    return "N/A" if x is None else f"{100*float(x):.2f}%"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ultra = load_json(ULTRA)
    core = load_json(OUT / "core_summary.json")
    random_label = load_json(OUT / "random_label_sanity_summary.json")
    robustness = load_json(OUT / "robustness_stress_summary.json")
    ood = load_json(OUT / "plantdoc_ood_summary.json")
    seeds = load_json(OUT / "efficientnet_seed_stability_summary.json")

    lines = [
        "# Full Control Validation Report",
        "",
        "This report summarizes post-audit checks. It does not claim field accuracy from PlantVillage.",
        "",
        "## 1. Split / leakage integrity",
    ]
    if ultra:
        lines += [
            f"- Ultra-strict train images: {ultra.get('train_examples')}",
            f"- Ultra-strict validation images: {ultra.get('validation_examples')}",
            f"- Locked official test images: {ultra.get('test_examples')}",
            f"- Strict dHash pairs before quarantine: {ultra.get('strict_dhash_pairs_before_quarantine')}",
            f"- Strict dHash pairs after quarantine: {ultra.get('strict_dhash_pairs_after_quarantine')}",
            f"- Quarantined train images: {ultra.get('quarantined_train_images')}",
            f"- Quarantined validation images: {ultra.get('quarantined_validation_images')}",
            f"- Quarantined test images: {ultra.get('quarantined_test_images')}",
            "",
        ]
    else:
        lines += ["- PENDING: ultra-strict summary not found.", ""]

    lines += ["## 2. Clean locked-test metrics and calibration"]
    if core:
        for row in core.get("metrics", []):
            lines.append(
                f"- {row['model']}: accuracy {pct(row['accuracy'])}, "
                f"Macro-F1 {row['macro_f1']:.4f}, ECE {row['ece_15_bins']:.4f}, "
                f"NLL {row['nll']:.4f}, errors {row['errors']}"
            )
        for row in core.get("bootstrap_95ci", []):
            lines.append(
                f"- {row['model']} 95% bootstrap accuracy CI: "
                f"{pct(row['accuracy_ci_low'])} to {pct(row['accuracy_ci_high'])}"
            )
        lines.append("")
    else:
        lines += ["- PENDING: run full_control_core.py.", ""]

    lines += ["## 3. Generalization gap"]
    gap_path = OUT / "generalization_gap.csv"
    if gap_path.is_file():
        gap = pd.read_csv(gap_path)
        for _, row in gap.iterrows():
            lines.append(
                f"- {row['model']}: clean-train {pct(row['clean_train_eval_accuracy'])}, "
                f"validation {pct(row['validation_accuracy'])}, test {pct(row['test_accuracy'])}, "
                f"validation-test gap {row['validation_minus_test_pp']:.2f} pp"
            )
        lines.append("")
    else:
        lines += ["- PENDING: generalization gap analysis not found.", ""]

    lines += ["## 4. Random-label sanity"]
    if random_label:
        lines += [
            f"- Result: {'PASS' if random_label.get('pass') else 'REVIEW'}",
            f"- Chance accuracy: {pct(random_label.get('chance_accuracy'))}",
            f"- True-label validation accuracy after random-label training: {pct(random_label.get('final_validation_accuracy'))}",
            f"- Validation Macro-F1: {random_label.get('final_validation_macro_f1'):.4f}",
            "",
        ]
    else:
        lines += ["- PENDING.", ""]

    lines += ["## 5. Robustness / shortcut stress"]
    if robustness:
        lines += [
            "- Completed on the locked test as a post-hoc stress test; it was not used for tuning.",
            "- See robustness_stress.csv and shortcut_occlusion_stress.csv for condition-by-condition drops.",
            "",
        ]
    else:
        lines += ["- PENDING.", ""]

    lines += ["## 6. External PlantDoc OOD probe"]
    if ood:
        for row in ood.get("results", []):
            lines.append(
                f"- {row['model']}: {pct(row['accuracy'])} accuracy, "
                f"mapped Macro-F1 {row['mapped_macro_f1']:.4f}, "
                f"{row['errors']}/{row['images']} errors"
            )
        lines += [
            "- Important: PlantDoc-to-PlantVillage labels are manually mapped and the datasets are not identical benchmarks.",
            "",
        ]
    else:
        lines += ["- PENDING.", ""]

    lines += ["## 7. Seed stability"]
    if seeds:
        lines += [
            f"- Seeds: {seeds.get('seeds')}",
            f"- Mean accuracy: {pct(seeds.get('accuracy_mean'))}",
            f"- Accuracy std: {100*float(seeds.get('accuracy_std')):.3f} pp",
            f"- Accuracy range: {seeds.get('accuracy_range_percentage_points'):.3f} pp",
            "",
        ]
    else:
        lines += ["- PENDING: seeds 123 and 777 have not both been completed.", ""]

    lines += [
        "## Interpretation boundary",
        "",
        "- These checks strengthen the claim that the high PlantVillage result is not explained by the audited leakage mechanisms alone.",
        "- External/OOD performance must be reported separately from PlantVillage accuracy.",
        "- Grad-CAM and occlusion tests are supporting shortcut-learning diagnostics, not proofs of causal feature use.",
        "- The locked test was not used for training, checkpoint selection, hyperparameter tuning, or ensemble-weight selection; its images were used for deterministic integrity/stress auditing only.",
        "",
    ]

    path = OUT / "FULL_CONTROL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
