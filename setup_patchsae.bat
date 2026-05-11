@echo off
REM Fix build issues and install PatchSAE requirements

cd /d d:\SAE\patchsae

echo.
echo ====================================
echo PATCHSAE SETUP - INSTALL BUILD TOOLS
echo ====================================
echo.

REM Pre-install critical packages as wheels to avoid build errors
echo Step 1: Pre-installing critical packages as wheels...
call ..\venv\Scripts\python.exe -m pip install --upgrade setuptools wheel
call ..\venv\Scripts\python.exe -m pip install --only-binary :all: numpy==1.26.4
call ..\venv\Scripts\python.exe -m pip install --only-binary :all: matplotlib==3.10.9
call ..\venv\Scripts\python.exe -m pip install --only-binary :all: scipy==1.14.1

echo.
echo Step 2: Installing PatchSAE requirements...
call ..\venv\Scripts\python.exe -m pip install -r requirements.txt --no-build-isolation

if errorlevel 1 (
    echo.
    echo WARNING: Some packages may have failed. Attempting to continue...
    echo.
)

echo.
echo ====================================
echo PatchSAE setup complete!
echo ====================================
echo.
pause

