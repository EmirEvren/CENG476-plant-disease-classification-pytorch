@echo off
setlocal

cd /d "%~dp0"
set PY=.\.venv\Scripts\python.exe

if not exist "%PY%" (
    echo [ERROR] Python virtual environment not found: %PY%
    exit /b 1
)

if not exist "outputs\audit\official_leaf_safe_split_manifest.csv" (
    echo [ERROR] Official leaf-safe manifest not found.
    echo Run: %PY% src\build_official_leaf_safe_manifest.py
    exit /b 1
)

if not exist "outputs\checkpoints\efficientnet_b0_official_leafsafe_full_best.pt" (
    echo [ERROR] Clean EfficientNet checkpoint not found.
    echo Train EfficientNet first with src\train_efficientnet_official.py
    exit /b 1
)

echo ============================================================
echo [1/6] Train Baseline CNN on official leaf-safe split
echo ============================================================
%PY% src\train_baseline_official.py --epochs 15 --batch-size 64 --num-workers 2 --learning-rate 5e-4 --weight-decay 1e-4 --dropout 0.4 --run-name baseline_cnn_official_leafsafe_full
if errorlevel 1 exit /b 1

echo ============================================================
echo [2/6] Evaluate Baseline CNN on locked official test
echo ============================================================
%PY% src\evaluate_baseline_official.py --checkpoint outputs\checkpoints\baseline_cnn_official_leafsafe_full_best.pt --batch-size 64
if errorlevel 1 exit /b 1

echo ============================================================
echo [3/6] Train ResNet18 on official leaf-safe split
echo ============================================================
%PY% src\train_resnet18_official.py --epochs 12 --batch-size 32 --num-workers 2 --backbone-learning-rate 1e-4 --classifier-learning-rate 5e-4 --weight-decay 1e-4 --dropout 0.3 --run-name resnet18_official_leafsafe_full
if errorlevel 1 exit /b 1

echo ============================================================
echo [4/6] Evaluate ResNet18 on locked official test
echo ============================================================
%PY% src\evaluate_resnet18_official.py --checkpoint outputs\checkpoints\resnet18_official_leafsafe_full_best.pt --batch-size 32
if errorlevel 1 exit /b 1

echo ============================================================
echo [5/6] Rebuild leaf-safe validation-tuned ensemble
echo ============================================================
%PY% src\evaluate_ensemble_official.py --resnet-checkpoint outputs\checkpoints\resnet18_official_leafsafe_full_best.pt --efficientnet-checkpoint outputs\checkpoints\efficientnet_b0_official_leafsafe_full_best.pt --batch-size 32
if errorlevel 1 exit /b 1

echo ============================================================
echo [6/6] Build final four-model comparison
echo ============================================================
%PY% src\compare_official_results.py
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo ALL REMAINING OFFICIAL LEAF-SAFE EXPERIMENTS COMPLETED
 echo ============================================================
echo EfficientNet-B0 clean result already present: 99.09%% accuracy
echo Final comparison: outputs\evaluation\official_leafsafe_final_comparison\final_model_comparison.csv
endlocal
