# Plant Disease Classification with PyTorch

**Course:** CENG 476 - Introduction to Deep Learning  
**Student:** Emir Evren  
**Student ID:** 210444038  
**Task:** 38-class plant disease classification on PlantVillage

This repository contains the complete PyTorch source code, experiment
configurations, training histories, test results, plots, ensemble analysis,
and Grad-CAM analysis used in the project report.

## Final Results

The test set was kept locked during training and model selection. Checkpoints
were selected only by validation macro-F1.

| Model | Parameters | Test accuracy | Test macro-F1 | Weighted-F1 |
|---|---:|---:|---:|---:|
| Baseline CNN | 399,142 | 87.42% | 0.8201 | 0.8711 |
| ResNet18 | 11,196,006 | 99.26% | 0.9869 | 0.9927 |
| EfficientNet-B0 | 4,056,226 | 99.52% | 0.9923 | 0.9952 |
| ResNet18 + EfficientNet-B0 ensemble | 15,252,232 | **99.76%** | **0.9950** | **0.9976** |

The final ensemble uses validation-selected 0.50 ResNet18 and 0.50
EfficientNet-B0 soft-voting weights.

## Project Structure

```text
.
|-- README.md
|-- requirements.txt
|-- data/
|   `-- raw/                       # Downloaded dataset (not included)
|-- notebooks/
|   `-- CENG476_Plant_Disease_Main_Experiments_Emir_Evren.ipynb
|-- src/
|   |-- download_data.py
|   |-- inspect_data.py
|   |-- data_setup.py
|   |-- visualize_data.py
|   |-- models.py
|   |-- sanity_check.py
|   |-- training.py
|   |-- train_baseline.py
|   |-- evaluate_baseline.py
|   |-- transfer_models.py
|   |-- train_resnet18.py
|   |-- evaluate_resnet18.py
|   |-- efficientnet_models.py
|   |-- train_efficientnet.py
|   |-- evaluate_efficientnet.py
|   |-- compare_models.py
|   |-- evaluate_ensemble.py
|   `-- generate_gradcam.py
|-- outputs/
|   |-- checkpoints/              # Generated checkpoints (not included)
|   |-- experiments/              # Saved hyperparameter configurations
|   |-- histories/                # Epoch-by-epoch training histories
|   |-- evaluation/               # Test metrics and predictions
|   |-- comparison/               # Final comparison tables
|   |-- gradcam/                  # Grad-CAM sample metadata
|   `-- figures/                  # Training/evaluation figures
`-- report/                       # Project report (DOCX and PDF)
```

## Main Experiment Notebook

The required Jupyter notebook is included at:

```text
notebooks/CENG476_Plant_Disease_Main_Experiments_Emir_Evren.ipynb
```

Open the project root in VS Code, open the notebook, select the
`.venv\Scripts\python.exe` kernel, and choose **Run All**. The notebook reads
the saved CSV, JSON, and PNG results, so the analysis sections do not retrain
the models. Live data-loader checks require the downloaded PlantVillage data;
optional live inference also requires the corresponding checkpoint.

## Dataset and Split

- Dataset: PlantVillage Kaggle mirror (`mohitsingh1804/plantvillage`)
- RGB images: 54,305
- Classes: 38
- Train set: 43,444 images
- Validation set: 5,430 images
- Locked test set: 5,431 images
- Split method: the dataset's original validation folder was divided
  50/50 using a stratified split with random seed 42.

The dataset is not included because of its size. After download, the expected
folders are:

```text
data/raw/PlantVillage/train/<class folders>
data/raw/PlantVillage/val/<class folders>
```

## Environment

The experiments were performed with:

- Windows
- Python 3.14.3
- PyTorch 2.11.0+cu130
- Torchvision 0.26.0+cu130
- NVIDIA GeForce RTX 3050 Laptop GPU (4 GB)
- CUDA 13.0 runtime supplied by the PyTorch wheel

The NVIDIA driver's displayed CUDA version does not have to exactly match the
PyTorch wheel's CUDA runtime. The driver only needs to be new enough to support
that runtime.

## Installation on Windows PowerShell

Open PowerShell in the project root.

### 1. Create the virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
```

### 2. Install CUDA-enabled PyTorch

The original environment used the following command:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu130
```

For a CPU-only installation, use the CPU wheel index instead:

```powershell
.\.venv\Scripts\python.exe -m pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install the remaining packages

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Verify the environment

```powershell
.\.venv\Scripts\python.exe -c "import torch, torchvision, pandas, sklearn, matplotlib, seaborn, kagglehub; print('Environment ready'); print('PyTorch:', torch.__version__); print('Torchvision:', torchvision.__version__); print('GPU available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## Download and Inspect the Data

Download the dataset:

```powershell
.\.venv\Scripts\python.exe src\download_data.py
```

Confirm the image counts and class mapping:

```powershell
.\.venv\Scripts\python.exe src\inspect_data.py
.\.venv\Scripts\python.exe src\data_setup.py
```

Generate the data visualizations:

```powershell
.\.venv\Scripts\python.exe src\visualize_data.py
```

## Sanity Check

Before full training, verify that the baseline model can overfit one image from
each class:

```powershell
.\.venv\Scripts\python.exe src\sanity_check.py
```

The expected result is a successful small-data overfitting test, demonstrating
that the forward pass, loss, backpropagation, and labels are connected
correctly.

## Reproduce the Main Training Runs

All commands below use the same names as the saved result files.

### Baseline CNN

```powershell
.\.venv\Scripts\python.exe src\train_baseline.py --epochs 15 --learning-rate 0.0005 --batch-size 64 --num-workers 2 --weight-decay 0.0001 --dropout 0.4 --run-name baseline_b64_lr5e4_full
```

### ResNet18

```powershell
.\.venv\Scripts\python.exe src\train_resnet18.py --epochs 12 --batch-size 32 --num-workers 2 --backbone-learning-rate 0.0001 --classifier-learning-rate 0.0005 --weight-decay 0.0001 --dropout 0.3 --run-name resnet18_b32_blr1e4_hlr5e4_full
```

### EfficientNet-B0

```powershell
.\.venv\Scripts\python.exe src\train_efficientnet.py --epochs 12 --batch-size 32 --num-workers 2 --backbone-learning-rate 0.0001 --classifier-learning-rate 0.0005 --weight-decay 0.0001 --dropout 0.3 --run-name efficientnet_b0_b32_blr1e4_hlr5e4_full
```

The first transfer-learning run downloads ImageNet pretrained weights from the
official PyTorch model repository.

## Evaluate the Locked Test Set

Only run these commands after training and checkpoint selection are complete.

```powershell
.\.venv\Scripts\python.exe src\evaluate_baseline.py --checkpoint outputs\checkpoints\baseline_b64_lr5e4_full_best.pt --batch-size 64

.\.venv\Scripts\python.exe src\evaluate_resnet18.py --checkpoint outputs\checkpoints\resnet18_b32_blr1e4_hlr5e4_full_best.pt --batch-size 32

.\.venv\Scripts\python.exe src\evaluate_efficientnet.py --checkpoint outputs\checkpoints\efficientnet_b0_b32_blr1e4_hlr5e4_full_best.pt --batch-size 32
```

Each evaluator saves the summary JSON, per-class report, predictions,
confusion matrices, top confusions, and plots under `outputs/`.

## Compare the Models

```powershell
.\.venv\Scripts\python.exe src\compare_models.py
```

This command reads the saved configurations, histories, and test summaries and
creates final comparison tables and plots.

## Ensemble Experiment

Place the best ResNet18 and EfficientNet-B0 checkpoints in
`outputs/checkpoints/`, then run:

```powershell
.\.venv\Scripts\python.exe src\evaluate_ensemble.py --batch-size 32
```

The ensemble searches candidate weights only on validation macro-F1. It then
evaluates the selected weights once on the locked test set. The test set is not
used to tune the weights.

## Grad-CAM Analysis

With the best EfficientNet-B0 checkpoint in `outputs/checkpoints/`, run:

```powershell
.\.venv\Scripts\python.exe src\generate_gradcam.py --batch-size 32 --num-correct 6 --num-errors 6
```

This produces separate figures for correctly and incorrectly classified test
images. For error cases, the visualization shows both the predicted-class and
true-class activation maps.

## Training Strategy

- Loss: unweighted cross-entropy
- Optimizer: AdamW
- Weight decay: 0.0001
- Scheduler: ReduceLROnPlateau monitored on validation loss
- Checkpoint selection: validation macro-F1
- Early stopping patience: 6 epochs for baseline, 5 for transfer models
- Mixed precision: enabled automatically when CUDA is available
- Random seed: 42
- Test set: never used during training, hyperparameter selection, scheduler
  updates, checkpoint selection, or ensemble weight selection

The transfer models use full fine-tuning with differential learning rates:
`1e-4` for the pretrained backbone and `5e-4` for the new classifier.

## Checkpoint Note

Model checkpoints are intentionally excluded from this package because they
are large generated artifacts (approximately 134 MB for ResNet18 and 49 MB for
EfficientNet-B0). Running the training commands recreates them under
`outputs/checkpoints/`.

## Windows Memory Troubleshooting

If Windows reports `WinError 1455` or says that the paging file is too small:

1. Close memory-heavy applications.
2. Confirm that no old Python training process is still running.
3. Retry with `--num-workers 0`.
4. If necessary, increase the Windows virtual-memory/page-file allocation and
   restart the computer.

Evaluation loaders already use zero worker processes to avoid repeatedly
loading PyTorch, SciPy, and CUDA DLLs in spawned Windows processes.

## Reproducibility Notes

- Random seeds are fixed where possible.
- CuDNN deterministic mode is enabled on CUDA.
- Exact experiment settings are saved in `outputs/experiments/`.
- Epoch metrics are saved in `outputs/histories/`.
- Complete predictions and per-class results are saved in
  `outputs/evaluation/`.
- Small numeric differences can still occur across operating systems, driver
  versions, GPU architectures, and library builds.
