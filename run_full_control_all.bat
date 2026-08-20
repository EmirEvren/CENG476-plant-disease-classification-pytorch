@echo off
setlocal
cd /d "%~dp0"

echo ============================================================
echo COMPLETE FULL CONTROL VALIDATION
echo ============================================================
echo Phase A: calibration, bootstrap, per-class, errors, robustness,
echo          random-label sanity, Grad-CAM, external PlantDoc OOD
echo Phase B: EfficientNet seeds 42 / 123 / 777
echo.
echo NOTE: Phase B retrains EfficientNet twice and can take several hours.
echo ============================================================

call run_full_control_fast.bat
if errorlevel 1 exit /b 1

call run_seed_stability.bat
if errorlevel 1 exit /b 1

echo ============================================================
echo COMPLETE FULL CONTROL FINISHED
echo ============================================================
echo Main audit folder: outputs\audit\full_control
echo Main figures folder: outputs\figures\full_control
endlocal
