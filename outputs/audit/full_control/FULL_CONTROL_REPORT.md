# Full Control Validation Report

This report summarizes post-audit checks. It does not claim field accuracy from PlantVillage.

## 1. Split / leakage integrity
- Ultra-strict train images: 39091
- Ultra-strict validation images: 4462
- Locked official test images: 10709
- Strict dHash pairs before quarantine: 39
- Strict dHash pairs after quarantine: 0
- Quarantined train images: 34
- Quarantined validation images: 4
- Quarantined test images: 0

## 2. Clean locked-test metrics and calibration
- EfficientNet-B0: accuracy 99.01%, Macro-F1 0.9874, ECE 0.0038, NLL 0.0357, errors 106
- 50/50 Ensemble: accuracy 99.14%, Macro-F1 0.9897, ECE 0.0091, NLL 0.0354, errors 92
- EfficientNet-B0 95% bootstrap accuracy CI: 98.81% to 99.20%
- 50/50 Ensemble 95% bootstrap accuracy CI: 98.95% to 99.31%

## 3. Generalization gap
- Baseline CNN: clean-train 77.89%, validation 83.55%, test 84.62%, validation-test gap -1.07 pp
- ResNet18: clean-train 98.16%, validation 98.81%, test 97.66%, validation-test gap 1.16 pp
- EfficientNet-B0: clean-train 99.74%, validation 98.68%, test 99.01%, validation-test gap -0.33 pp

## 4. Random-label sanity
- Result: PASS
- Chance accuracy: 2.63%
- True-label validation accuracy after random-label training: 1.84%
- Validation Macro-F1: 0.0183

## 5. Robustness / shortcut stress
- Completed on the locked test as a post-hoc stress test; it was not used for tuning.
- See robustness_stress.csv and shortcut_occlusion_stress.csv for condition-by-condition drops.

## 6. External PlantDoc OOD probe
- EfficientNet-B0: 23.31% accuracy, mapped Macro-F1 0.2183, 181/236 errors
- 50/50 Ensemble: 25.00% accuracy, mapped Macro-F1 0.2349, 177/236 errors
- Important: PlantDoc-to-PlantVillage labels are manually mapped and the datasets are not identical benchmarks.

## 7. Seed stability
- Seeds: [42, 123, 777]
- Mean accuracy: 99.00%
- Accuracy std: 0.229 pp
- Accuracy range: 0.458 pp

## Interpretation boundary

- These checks strengthen the claim that the high PlantVillage result is not explained by the audited leakage mechanisms alone.
- External/OOD performance must be reported separately from PlantVillage accuracy.
- Grad-CAM and occlusion tests are supporting shortcut-learning diagnostics, not proofs of causal feature use.
- The locked test was not used for training, checkpoint selection, hyperparameter tuning, or ensemble-weight selection; its images were used for deterministic integrity/stress auditing only.
