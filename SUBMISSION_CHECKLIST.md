# Submission Checklist

This checklist reflects the actual repository state. Items are checked only when the corresponding artifact or experiment result is present in the repository.

## Required Deliverables

- [ ] Final project report in PDF format
- [ ] Editable project report in DOCX format (recommended working copy)
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
- [x] Batch normalization placement documented
- [x] Dropout placement and selected rates documented
- [ ] Dropout-rate ablation completed for 0.2 / 0.4 / 0.6 and compared
- [x] Weight decay documented (`1e-4`)
- [x] Image data augmentation documented
- [x] Training/validation curves available
- [x] Overfitting/underfitting discussion supported by clean-train and validation metrics
- [x] AdamW optimizer and learning rates documented
- [x] Learning-rate pilot experiments completed
- [x] ReduceLROnPlateau scheduler implemented and observed in training histories
- [x] Early stopping implemented using validation Macro-F1
- [ ] Early-stopping demonstration run added only if needed for presentation evidence
- [x] Activation functions and raw-logit/CrossEntropyLoss design documented
- [x] Dataset source, sample count, dimensions, class count, and imbalance documented
- [x] Stratified train/validation/test protocol documented
- [x] Accuracy, precision, recall, F1, ROC-AUC, per-class results, and confusion matrices computed
- [ ] Macro/micro ROC curve figure generated for the final individual model
- [x] Error analysis included
- [x] Ensemble experiment included
- [x] Grad-CAM experiment included
- [ ] Baseline architecture diagram generated and committed
- [ ] Absolute local Windows paths removed from tracked JSON outputs

## Intentionally Excluded

- PlantVillage image files, because they can be downloaded with `src/download_data.py`
- `.venv`, cache folders, and compiled Python files
- Large generated model checkpoints; training commands recreate them

## Before Uploading

1. Run the remaining assignment-completeness commands documented in `ASSIGNMENT_FIX_PLAN.md`.
2. Open the final PDF report and confirm the student name and ID.
3. Confirm `README.md` and the report describe only experiments that actually have saved evidence.
4. Keep the project folder name unchanged.
5. Do not upload `.venv` or the downloaded dataset.
6. If checkpoint files are required separately, use Git LFS or another large-file service and document the links.

## Submission and Presentation Dates

- Code: 24 August 2026 before the presentation
- Public GitHub link by email: 25 August 2026
- Report and code through Teams: 28 August 2026 at 23:59

Prepare a clear **10-minute presentation plus 8 minutes of questions** covering:

1. Problem and PlantVillage dataset
2. Scratch Baseline CNN and transfer-learning comparison
3. Regularization and training strategy
4. Learning-rate and dropout experiments
5. Main test results and error analysis
6. Ensemble and Grad-CAM
7. Limitations, lessons learned, and future work
