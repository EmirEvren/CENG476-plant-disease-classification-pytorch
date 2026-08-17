# CENG 476 Assignment Completion Plan

This file closes the remaining gaps between the repository and the assignment requirements.

## 1. Dropout ablation

The assignment explicitly asks whether different dropout values were tried and what effect they had. Keep every other baseline setting fixed and run only the dropout value as the independent variable.

```powershell
.\.venv\Scripts\python.exe src\train_baseline.py --epochs 3 --learning-rate 0.0005 --batch-size 64 --num-workers 2 --weight-decay 0.0001 --dropout 0.2 --run-name baseline_dropout02_pilot

.\.venv\Scripts\python.exe src\train_baseline.py --epochs 3 --learning-rate 0.0005 --batch-size 64 --num-workers 2 --weight-decay 0.0001 --dropout 0.6 --run-name baseline_dropout06_pilot

.\.venv\Scripts\python.exe src\analyze_dropout_experiments.py
```

The existing `baseline_b64_lr5e4_pilot` run is used as the dropout `0.4` reference. The analysis command creates:

- `outputs/comparison/dropout_ablation.csv`
- `outputs/figures/dropout_ablation_validation_macro_f1.png`

Do not use the test set to choose the dropout value.

## 2. Architecture diagram

Generate a high-level figure for the custom CNN:

```powershell
.\.venv\Scripts\python.exe src\generate_architecture_diagram.py
```

Output:

- `outputs/figures/baseline_cnn_architecture.png`

## 3. ROC curve figure

The evaluation scripts already compute ROC-AUC. Generate a readable macro/micro one-vs-rest ROC figure for the strongest individual model (EfficientNet-B0):

```powershell
.\.venv\Scripts\python.exe src\generate_roc_curves.py --checkpoint outputs\checkpoints\efficientnet_b0_b32_blr1e4_hlr5e4_full_best.pt --batch-size 32
```

Output:

- `outputs/figures/efficientnet_b0_b32_blr1e4_hlr5e4_full_test_roc_curve.png`

The 38 per-class ROC curves are intentionally not placed on one figure because that would be unreadable. Per-class ROC-AUC values remain available in the classification report.

## 4. Portable output paths

Remove machine-specific absolute Windows paths from tracked JSON result files:

```powershell
.\.venv\Scripts\python.exe src\sanitize_output_paths.py
```

After running it, inspect `git diff` before committing. Only path strings inside `outputs/` should become repository-relative.

## 5. Early stopping wording

Early stopping is already implemented and monitors validation Macro-F1. The final runs reached their configured maximum epochs before the patience threshold stopped them. The report must state this accurately instead of claiming that early stopping terminated those runs.

Recommended report wording:

> Patience-based early stopping monitored validation Macro-F1 and acted as a safeguard against unnecessary training. In the final reported runs, the configured maximum epoch limit was reached before the patience criterion triggered, so early stopping did not terminate those runs. Best-checkpoint selection remained independent and preserved the epoch with the strongest validation Macro-F1.

A separate forced early-stopping demonstration is optional and should not replace the final experiments.

## 6. Optimizer details to report

For the custom CNN, report the complete AdamW setup:

- optimizer: AdamW
- beta1: 0.9
- beta2: 0.999
- weight decay: `1e-4`
- selected baseline learning rate: `5e-4`
- loss: unweighted cross-entropy

For transfer learning, report the differential learning rates:

- pretrained backbone: `1e-4`
- new classifier: `5e-4`
- weight decay: `1e-4`

## 7. Softmax explanation

All models intentionally return raw logits. Do not add a Softmax layer before `CrossEntropyLoss`.

Report/presentation explanation:

> PyTorch `CrossEntropyLoss` expects raw logits and internally applies the log-softmax operation required for multi-class classification. Explicit Softmax is therefore used only during inference/evaluation when class probabilities are needed.

## 8. Final report

Create the final report only after the dropout ablation, architecture figure, ROC figure, and portable-path cleanup are complete.

Required report structure:

1. Introduction
2. Related Work / Baseline
3. Proposed Method
   - Baseline CNN
   - ResNet18 transfer learning
   - EfficientNet-B0 transfer learning
   - Activations and output design
   - Regularization
4. Experimental Setup
   - Dataset and split
   - Preprocessing and augmentation
   - Optimizer and hyperparameters
   - LR tuning and scheduler
   - Early stopping
5. Results and Discussion
   - Learning-rate pilots
   - Dropout ablation
   - Training/validation curves
   - Test metrics
   - Per-class analysis and confusion matrix
   - ROC/AUC
   - Scratch vs transfer learning
   - Overfitting/underfitting analysis
   - Ensemble
   - Grad-CAM
6. Limitations and Future Work
7. Conclusion
8. References

## 9. Final validation

Before submission:

```powershell
.\.venv\Scripts\python.exe -m py_compile src\*.py
.\.venv\Scripts\python.exe src\data_setup.py
.\.venv\Scripts\python.exe src\compare_models.py
```

Then verify that every checked item in `SUBMISSION_CHECKLIST.md` has a corresponding file, table, plot, source-code implementation, or saved experiment output.
