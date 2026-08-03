# Submission Checklist

## Required Deliverables

- [x] Project report in PDF format
- [x] Editable project report in DOCX format
- [x] Well-organized PyTorch source code
- [x] Installation and execution instructions
- [x] Jupyter notebook demonstrating the main experiments
- [x] Fixed random seed and documented data split
- [x] Saved experiment configurations and training histories
- [x] Test metrics, per-class reports, confusion matrices, and predictions
- [x] Creative experiment: validation-tuned soft-voting ensemble
- [x] Explainability experiment: Grad-CAM

## Intentionally Excluded

- PlantVillage image files, because they can be downloaded with
  `src/download_data.py`
- `.venv`, cache folders, and compiled Python files
- Large generated model checkpoints; training commands recreate them

## Before Uploading

1. Open the PDF report and confirm the student name and ID.
2. Keep the project folder name unchanged.
3. If publishing on GitHub, upload the extracted project folder rather than
   the outer ZIP.
4. Confirm that `README.md` is visible on the repository front page.
5. Do not upload the `.venv` or downloaded dataset.
6. If checkpoint files are required separately, upload them using a
   large-file service or Git LFS and add the links to `README.md`.

## Remaining Submission and Presentation Steps

1. Publish the extracted project folder in a public GitHub repository.
2. Email the public GitHub link to `mkbinli@thk.edu.tr` by 25 August 2026.
3. Submit the code by 24 August 2026 before the presentation.
4. Submit the final report and code through Teams by 28 August 2026 at 23:59.

Prepare a clear **10-minute presentation plus 8 minutes of questions** covering:

1. Problem and PlantVillage dataset
2. Baseline CNN, ResNet18, and EfficientNet-B0
3. Regularization and training strategy
4. Main test results and model comparison
5. Ensemble and Grad-CAM
6. Limitations, lessons learned, and future work
