@echo off
setlocal
cd /d "%~dp0"

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" (
  echo [ERROR] Virtual environment Python not found: %PYTHON%
  exit /b 1
)

echo ============================================================
echo EFFICIENTNET ULTRA-STRICT SEED STABILITY
echo Fixed manifest, same locked 10,709-image test
echo ============================================================

set "RUN123=efficientnet_b0_official_ultrastrict_seed123"
set "RUN777=efficientnet_b0_official_ultrastrict_seed777"

if not exist "outputs\checkpoints\%RUN123%_best.pt" (
  echo [1/6] Train seed 123
  "%PYTHON%" src\train_efficientnet_ultrastrict_seed.py --seed 123 --run-name %RUN123% --epochs 12 --batch-size 32 --num-workers 2
  if errorlevel 1 exit /b 1
) else (
  echo [1/6] Seed 123 checkpoint already exists - skipping training.
)

if not exist "outputs\evaluation\%RUN123%\test_summary.json" (
  echo [2/6] Evaluate seed 123 on unchanged locked test
  "%PYTHON%" src\ultrastrict_entrypoint.py evaluate_efficientnet_official --checkpoint outputs\checkpoints\%RUN123%_best.pt --batch-size 32
  if errorlevel 1 exit /b 1
) else (
  echo [2/6] Seed 123 evaluation already exists - skipping.
)

if not exist "outputs\checkpoints\%RUN777%_best.pt" (
  echo [3/6] Train seed 777
  "%PYTHON%" src\train_efficientnet_ultrastrict_seed.py --seed 777 --run-name %RUN777% --epochs 12 --batch-size 32 --num-workers 2
  if errorlevel 1 exit /b 1
) else (
  echo [3/6] Seed 777 checkpoint already exists - skipping training.
)

if not exist "outputs\evaluation\%RUN777%\test_summary.json" (
  echo [4/6] Evaluate seed 777 on unchanged locked test
  "%PYTHON%" src\ultrastrict_entrypoint.py evaluate_efficientnet_official --checkpoint outputs\checkpoints\%RUN777%_best.pt --batch-size 32
  if errorlevel 1 exit /b 1
) else (
  echo [4/6] Seed 777 evaluation already exists - skipping.
)

echo [5/6] Summarize seeds 42, 123, 777
"%PYTHON%" src\summarize_seed_stability.py
if errorlevel 1 exit /b 1

echo [6/6] Rebuild consolidated report
"%PYTHON%" src\build_full_control_report.py
if errorlevel 1 exit /b 1

echo ============================================================
echo SEED STABILITY COMPLETED
echo ============================================================
echo Summary: outputs\audit\full_control\efficientnet_seed_stability_summary.json
echo Report: outputs\audit\full_control\FULL_CONTROL_REPORT.md
endlocal
