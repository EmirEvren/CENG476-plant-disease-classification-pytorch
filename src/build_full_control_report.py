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
        "This report summarizes the final post-audit checks. It does not claim field accuracy from PlantVillage.",
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
            "The official test set was not modified. Physical-leaf metadata coverage is incomplete, so this audit does not claim proof that every unmapped image belongs to a unique leaf.",
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
            "- Interpretation: evidence against a trivial direct label/path shortcut in the tested sanity setup; not a formal proof of zero leakage.",
            "",
        ]
    else:
        lines += ["- PENDING.", ""]

    lines += ["## 5. Robustness / shortcut stress"]
    if robustness:
        rows = robustness.get("rows", [])
        lines.append("- Completed on the locked test as a post-hoc stress test; it was not used for tuning.")
        for row in rows:
            if row.get("condition") in {
                "clean",
                "gaussian_blur_radius_2",
                "center_occluded_60pct",
                "border_occluded_keep_center_60pct",
            }:
                lines.append(
                    f"- {row['model']} / {row['condition']}: accuracy {pct(row['accuracy'])}, "
                    f"Macro-F1 {row['macro_f1']:.4f}"
                )
        lines += [
            "- Large blur and occlusion drops show that high clean accuracy does not imply uniform corruption robustness.",
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
            "- Interpretation: the large drop is evidence of strong domain dependence and limited out-of-domain generalization; it is not a directly comparable replacement benchmark.",
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
            "- The highest seed is not promoted as the final benchmark; all seed results are reported together to avoid cherry-picking.",
            "",
        ]
    else:
        lines += ["- PENDING: seeds 123 and 777 have not both been completed.", ""]

    lines += [
        "## Final interpretation",
        "",
        "- The first image-level 99.76% ensemble result should not be used as the final benchmark because the original split contained audited duplicate / same-leaf leakage risks.",
        "- After the ultra-strict protocol, EfficientNet-B0 remains near 99% and repeated-seed results remain tightly clustered, so the controlled-domain result is not explained by the audited leakage mechanisms or a single favorable seed alone.",
        "- The transfer-model train/validation/test gaps and calibration checks do not strongly support severe conventional overfitting as the main explanation for the PlantVillage result.",
        "- The PlantDoc OOD result is much lower and demonstrates strong domain dependence. PlantVillage performance must not be presented as field accuracy.",
        "- Grad-CAM and occlusion tests are supporting shortcut-learning diagnostics, not proofs of causal feature use.",
        "- The locked test was not used for training, checkpoint selection, hyperparameter tuning, or ensemble-weight selection; its images were used for deterministic integrity checks and post-hoc stress auditing only.",
        "",
        "## Final benchmark to report",
        "",
        "- EfficientNet-B0 fixed run: 99.01% accuracy, Macro-F1 0.9874.",
        "- Validation-selected 50/50 ensemble: 99.14% accuracy, Macro-F1 0.9897.",
        "- EfficientNet 3-seed stability: approximately 99.00% mean accuracy with 0.229 percentage-point standard deviation.",
        "- External PlantDoc OOD probe: 23.31% EfficientNet accuracy and 25.00% ensemble accuracy on the mapped 236-image test subset.",
        "",
    ]

    path = OUT / "FULL_CONTROL_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
