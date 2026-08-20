# Final Validation Summary

This document is the concise methodological summary for the final CENG476 project result.

## Final reported benchmark

The original image-level split produced a 99.76% ResNet18 + EfficientNet-B0 ensemble accuracy. A later integrity audit found cross-split duplicate / same-physical-leaf contamination, so that value is retained only as a historical result and is **not** used as the final benchmark.

The final benchmark is based on an ultra-strict PlantVillage protocol with a fixed official test split and additional duplicate / near-duplicate controls.

| Model | Test Accuracy | Macro-F1 | Errors / 10,709 |
|---|---:|---:|---:|
| Custom CNN | 84.62% | 0.7813 | 1,647 |
| ResNet18 | 97.66% | 0.9686 | 251 |
| EfficientNet-B0 | **99.01%** | **0.9874** | **106** |
| ResNet18 + EfficientNet-B0 | **99.14%** | **0.9897** | **92** |

The fixed 50/50 ensemble weight was selected from validation performance only.

## Split-integrity result

Final ultra-strict split:

- train: 39,091 images,
- validation: 4,462 images,
- locked official test: 10,709 images,
- 38 classes.

Additional integrity controls:

- exact cross-split collisions removed from the training side,
- mapped physical-leaf train/test overlap: 0,
- mapped physical-leaf train/validation overlap: 0,
- 39 strict `dHash <= 4` cross-split pairs found before quarantine,
- 34 train and 4 validation images quarantined,
- 0 test images removed,
- strict `dHash <= 4` cross-split pairs after quarantine: 0.

The test set itself was not changed. Test images were used in deterministic integrity auditing and later post-hoc stress analysis, but the test set was not used for training, checkpoint selection, hyperparameter tuning, or ensemble-weight selection.

Because PlantVillage leaf-ID coverage is incomplete, this protocol does not prove that every unmapped image corresponds to a unique physical leaf. The supported claim is limited to the overlap mechanisms explicitly audited.

## Overfitting / reproducibility controls

### Generalization gaps

At the selected checkpoints:

- ResNet18: clean-train 98.16%, validation 98.81%, test 97.66%.
- EfficientNet-B0: clean-train 99.74%, validation 98.68%, test 99.01%.

These values do not show the large validation/test collapse expected from severe conventional overfitting in the transfer models.

### Seed stability

EfficientNet-B0 was retrained on the same fixed ultra-strict manifest:

| Seed | Test Accuracy | Test Macro-F1 |
|---:|---:|---:|
| 42 | 99.010% | 0.987373 |
| 123 | 98.767% | 0.983668 |
| 777 | 99.225% | 0.990149 |

- mean accuracy: 99.001%,
- standard deviation: 0.229 percentage points,
- total range: 0.458 percentage points.

The seed-777 run is not promoted as the final result simply because it is the highest result. The original fixed run and the 3-seed summary are reported separately to avoid cherry-picking.

### Calibration

- EfficientNet-B0 ECE (15 bins): 0.003845,
- ensemble ECE (15 bins): 0.009121.

Bootstrap 95% accuracy confidence intervals:

- EfficientNet-B0: 98.81% to 99.20%,
- ensemble: 98.95% to 99.31%.

### Random-label sanity test

With shuffled training labels on a balanced subset:

- 38-class chance level: 2.63%,
- true-label validation accuracy after random-label training: 1.84%,
- Macro-F1: 0.0183,
- sanity result: PASS.

This is evidence against a trivial label/path leakage shortcut in the tested pipeline, not a formal proof of zero leakage.

## Robustness findings

Selected post-hoc stress results:

| Condition | EfficientNet-B0 | Ensemble |
|---|---:|---:|
| Clean | 99.01% | 99.14% |
| Brightness x0.60 | 98.83% | 99.18% |
| Brightness x1.40 | 98.07% | 98.73% |
| Contrast x0.60 | 98.49% | 98.79% |
| JPEG quality 30 | 98.13% | 98.72% |
| Gaussian blur radius 2 | 84.08% | 86.53% |
| Rotation 15 degrees | 99.41% | 99.51% |

Large occlusion also causes major degradation. The models are therefore highly accurate in-domain but not uniformly robust to information loss.

Grad-CAM visualizations are used only as supporting qualitative evidence and are not treated as proof of causal attention.

## External domain result

A mapped 236-image subset of the PlantDoc test split was used as an out-of-domain stress test without retraining.

- EfficientNet-B0: 23.31% accuracy, mapped Macro-F1 0.2183,
- 50/50 ensemble: 25.00% accuracy, mapped Macro-F1 0.2349.

PlantDoc and PlantVillage are not identical benchmarks and the compatible labels are manually mapped. This result must therefore be interpreted as a **domain-shift probe**, not a directly comparable test replacement.

The large performance drop is still scientifically important: it shows that the near-99% PlantVillage result does not imply equivalent real-world / field performance.

## Final conclusion

The final evidence supports the following interpretation:

1. The first 99.76% result came from an optimistic image-level split that contained leakage risks and should not be used as the final benchmark.
2. After a stricter leaf-aware and near-duplicate-controlled protocol, near-99% accuracy remains reproducible inside PlantVillage.
3. Repeated-seed, calibration, random-label and train/validation/test checks do not strongly support severe conventional overfitting as the main explanation for the controlled-domain result.
4. External PlantDoc performance demonstrates strong domain dependence and limited out-of-domain generalization.

**Final benchmark to report:**

- EfficientNet-B0: **99.01% accuracy, 0.9874 Macro-F1**.
- Validation-selected ensemble: **99.14% accuracy, 0.9897 Macro-F1**.
- EfficientNet seed stability: **99.00% mean accuracy, 0.229 pp standard deviation across 3 seeds**.
- External PlantDoc OOD probe: **23.31% / 25.00% accuracy for EfficientNet / ensemble**.
