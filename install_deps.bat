@echo off
REM Install all dependencies in the SAE venv
REM This script installs PyTorch with CUDA 12.8 support and all ML libraries

cd /d d:\SAE

echo.
echo ====================================
echo SAE PROJECT - DEPENDENCY INSTALLATION
echo ====================================
echo.
echo Python: %cd%\venv\Scripts\python.exe
echo.

REM Upgrade pip (use python -m pip to avoid conflicts on Windows)
echo Step 1: Upgrading pip...
call venv\Scripts\python.exe -m pip install --upgrade pip
if errorlevel 1 (
    echo ERROR: pip upgrade failed
    exit /b 1
)

REM Install PyTorch with CUDA 12.8
echo.
echo Step 2: Installing PyTorch with CUDA 12.8 support...
echo This will download ~2-3 GB, please be patient...
call venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo ERROR: PyTorch installation failed
    exit /b 1
)

REM Install core ML libraries
echo.
echo Step 3: Installing core ML libraries...
call venv\Scripts\python.exe -m pip install ^
    open_clip_torch ^
    transformers ^
    datasets ^
    matplotlib ^
    pandas ^
    scikit-learn ^
    einops ^
    tqdm ^
    jupyter ^
    scipy
if errorlevel 1 (
    echo ERROR: Core libraries installation failed
    exit /b 1
)

REM Verify installation
echo.
echo Step 4: Verifying installation...
call venv\Scripts\python.exe -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo.
echo ====================================
echo Installation complete!
echo Next: Download SAE checkpoints
echo ====================================
echo.
pause
