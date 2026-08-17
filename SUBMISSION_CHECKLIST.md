# Submission Checklist

This checklist reflects the actual project evidence after the assignment-completeness pass.

## Required Deliverables

- [ ] Final project report PDF added to the repository/submission package
- [ ] Editable project report DOCX added to the repository/submission package
- [x] Well-organized PyTorch source code
- [x] Installation and execution instructions
- [x] Jupyter notebook demonstrating the main experiments
- [x] Fixed random seed and documented data split
- [x] Saved experiment configurations and training histories
- [x] Test metrics, per-class reports, confusion matrices, and predictions
- [x] Creative experiment: validation-tuned soft-voting ensemble
- [x] Explainability experiment: Grad-CAM

## Assignment-Requirement Completion

- [x] Custom CNN architecture trained from scratch
- [x] Transfer learning with ResNet18 and EfficientNet-B0
- [x] Input/output shapes and main architecture blocks documented
- [x] Batch Normalization placement documented
- [x] Dropout placement and selected rates documented
- [x] Controlled dropout-rate ablation completed for `0.2 / 0.4 / 0.6`
- [x] Dropout ablation CSV and validation Macro-F1 figure committed
- [x] Weight decay documented (`1e-4`)
- [x] Image data augmentation documented
- [x] Training/validation curves available
- [x] Overfitting/underfitting discussion supported by clean-train and validation metrics
- [x] AdamW optimizer, betas, and learning rates documented
- [x] Learning-rate pilot experiments completed
- [x] ReduceLROnPlateau scheduler implemented and observed in training histories
- [x] Early stopping implemented using validation Macro-F1
- [x] Final-run early-stopping behavior accurately documented: maximum epoch limits were reached before patience was exhausted
- [x] Activation functions and raw-logit/CrossEntropyLoss design documented
- [x] Dataset source, sample count, dimensions, class count, and imbalance documented
- [x] Stratified train/validation/test protocol documented
- [x] Accuracy, precision, recall, F1, ROC-AUC, per-class results, and confusion matrices computed
- [x] Macro/micro ROC curve generated for EfficientNet-B0
- [x] Error analysis included
- [x] Ensemble experiment included
- [x] Grad-CAM experiment included
- [x] Baseline CNN architecture diagram generated and committed
- [x] Absolute local Windows paths removed from tracked JSON outputs
- [x] README updated to match the completed experiments

## Final Evidence Added

- `outputs/comparison/dropout_ablation.csv`
- `outputs/figures/dropout_ablation_validation_macro_f1.png`
- `outputs/figures/baseline_cnn_architecture.png`
- `outputs/figures/efficientnet_b0_b32_blr1e4_hlr5e4_full_test_roc_curve.png`
- `outputs/histories/baseline_dropout02_pilot_history.csv`
- `outputs/histories/baseline_dropout06_pilot_history.csv`
- sanitized evaluation and Grad-CAM JSON summaries

## Intentionally Excluded

- PlantVillage image files, because they can be recreated with `src/download_data.py`
- `.venv`, cache folders, and compiled Python files
- large generated checkpoints; training commands recreate them

## Remaining Submission Action

Only the final report files still need to be placed in the repository/submission package:

- `report/CENG476_Plant_Disease_Classification_Report_Emir_Evren.pdf`
- `report/CENG476_Plant_Disease_Classification_Report_Emir_Evren.docx`

After adding those files, open the PDF once locally and confirm the student name/ID and all figures render correctly.

## Submission and Presentation Dates

- Code: 24 August 2026 before the presentation
- Public GitHub link by email: 25 August 2026
- Report and code through Teams: 28 August 2026 at 23:59

Presentation target: **10 minutes + 8 minutes Q&A**.
