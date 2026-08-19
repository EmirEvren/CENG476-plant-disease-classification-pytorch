@echo off
setlocal

cd /d "%~dp0"
set PY=.\.venv\Scripts\python.exe

if not exist "%PY%" (
    echo [ERROR] Python virtual environment not found: %PY%
    exit /b 1
)

if not exist "outputs\audit\final_leafsafe_protocol_audit\dhash_strict_cross_split_pairs.csv" (
    echo [ERROR] Strict dHash audit CSV not found.
    echo Run run_final_leafsafe_audit.bat first.
    exit /b 1
)

echo ============================================================
echo [1/10] Build ultra-strict quarantine manifest
 echo ============================================================
%PY% src\build_ultrastrict_manifest.py
if errorlevel 1 exit /b 1

echo ============================================================
echo [2/10] Train Baseline CNN - ultra-strict
 echo ============================================================
%PY% src\ultrastrict_entrypoint.py train_baseline_official --epochs 15 --batch-size 64 --num-workers 2 --learning-rate 5e-4 --weight-decay 1e-4 --dropout 0.4 --run-name baseline_cnn_official_ultrastrict_full
if errorlevel 1 exit /b 1

echo ============================================================
echo [3/10] Evaluate Baseline CNN - unchanged locked test
 echo ============================================================
%PY% src\ultrastrict_entrypoint.py evaluate_baseline_official --checkpoint outputs\checkpoints\baseline_cnn_official_ultrastrict_full_best.pt --batch-size 64
if errorlevel 1 exit /b 1

echo ============================================================
echo [4/10] Train ResNet18 - ultra-strict
 echo ============================================================
%PY% src\ultrastrict_entrypoint.py train_resnet18_official --epochs 12 --batch-size 32 --num-workers 2 --backbone-learning-rate 1e-4 --classifier-learning-rate 5e-4 --weight-decay 1e-4 --dropout 0.3 --run-name resnet18_official_ultrastrict_full
if errorlevel 1 exit /b 1

echo ============================================================
echo [5/10] Evaluate ResNet18 - unchanged locked test
 echo ============================================================
%PY% src\ultrastrict_entrypoint.py evaluate_resnet18_official --checkpoint outputs\checkpoints\resnet18_official_ultrastrict_full_best.pt --batch-size 32
if errorlevel 1 exit /b 1

echo ============================================================
echo [6/10] Train EfficientNet-B0 - ultra-strict
 echo ============================================================
%PY% src\ultrastrict_entrypoint.py train_efficientnet_official --epochs 12 --batch-size 32 --num-workers 2 --backbone-learning-rate 1e-4 --classifier-learning-rate 5e-4 --weight-decay 1e-4 --dropout 0.3 --run-name efficientnet_b0_official_ultrastrict_full
if errorlevel 1 exit /b 1

echo ============================================================
echo [7/10] Evaluate EfficientNet-B0 - unchanged locked test
 echo ============================================================
%PY% src\ultrastrict_entrypoint.py evaluate_efficientnet_official --checkpoint outputs\checkpoints\efficientnet_b0_official_ultrastrict_full_best.pt --batch-size 32
if errorlevel 1 exit /b 1

echo ============================================================
echo [8/10] Select ensemble weights on ultra-strict validation only
 echo ============================================================
%PY% src\ultrastrict_entrypoint.py evaluate_ensemble_official --resnet-checkpoint outputs\checkpoints\resnet18_official_ultrastrict_full_best.pt --efficientnet-checkpoint outputs\checkpoints\efficientnet_b0_official_ultrastrict_full_best.pt --batch-size 32
if errorlevel 1 exit /b 1

echo ============================================================
echo [9/10] Build final ultra-strict comparison
 echo ============================================================
%PY% src\compare_ultrastrict_results.py
if errorlevel 1 exit /b 1

echo ============================================================
echo [10/10] Complete
 echo ============================================================
echo Ultra-strict manifest:
echo outputs\audit\official_leaf_safe_split_manifest_ultrastrict.csv
echo Final comparison:
echo outputs\evaluation\official_ultrastrict_final_comparison\final_model_comparison.csv
echo.
echo The 10,709-image official test was never modified.
endlocal
