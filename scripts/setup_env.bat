@echo off
REM ============================================================
REM F-Zero RL - Complete Environment Setup Script
REM
REM HOW TO RUN:
REM   1. Open Start Menu, search "x64 Native Tools Command Prompt for VS 2019" (or 2022)
REM   2. In that prompt, run:
REM        D:\Anaconda3\Scripts\activate.bat fzero
REM        "d:\Code\AI Project\RL\RL-Gaming\scripts\setup_env.bat"
REM ============================================================

setlocal enabledelayedexpansion

set SCRIPT_DIR=%~dp0
set PROJECT_DIR=%SCRIPT_DIR%..

echo ============================================================
echo  F-Zero RL Environment Setup
echo ============================================================
echo.

REM --- Check Python ---
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Run this first:
    echo   D:\Anaconda3\Scripts\activate.bat fzero
    exit /b 1
)
echo [OK] Python found:
python --version

REM --- Check cl.exe exists ---
where cl.exe >nul 2>&1
if errorlevel 1 (
    echo ERROR: cl.exe not found.
    echo Open "x64 Native Tools Command Prompt for VS 2022" instead.
    exit /b 1
)

REM --- Check cl.exe is 64-bit (critical!) ---
REM Check for "x64" or "x86_amd64" in cl output (works for EN and CN locale)
cl 2>&1 | findstr /i /c:"x64" /c:"amd64" >nul
if errorlevel 1 (
    echo ERROR: 32-bit compiler detected. You need the 64-bit compiler.
    echo Open "x64 Native Tools Command Prompt for VS 2019/2022" instead
    echo of the regular Developer Command Prompt.
    echo.
    echo Or run this in your current prompt to switch to x64:
    echo   "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    echo   "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    exit /b 1
)
echo [OK] 64-bit C++ compiler found

REM --- Set UTF-8 encoding ---
set PYTHONUTF8=1
echo [OK] UTF-8 encoding set
echo.

REM --- Install build dependencies ---
echo [1/6] Installing build dependencies...
call conda install zlib m2w64-toolchain libpython -y -q 2>nul
echo [OK] build dependencies ready

REM --- Verify prerequisites ---
echo.
echo [2/6] Checking prerequisites...
python "%SCRIPT_DIR%setup_helper.py" verify
echo.

REM --- Clone stable-retro ---
echo [3/6] Cloning stable-retro source...
set RETRO_BUILD_DIR=%TEMP%\stable-retro-build
if exist "%RETRO_BUILD_DIR%" (
    echo   Cleaning previous build...
    rmdir /s /q "%RETRO_BUILD_DIR%" 2>nul
    timeout /t 2 /nobreak >nul
)
git clone --depth 1 https://github.com/Farama-Foundation/stable-retro.git "%RETRO_BUILD_DIR%"
if errorlevel 1 (
    echo ERROR: git clone failed
    exit /b 1
)
echo [OK] Source cloned

REM --- Patch setup.py ---
echo.
echo [4/6] Patching setup.py for Windows...
python "%SCRIPT_DIR%setup_helper.py" patch "%RETRO_BUILD_DIR%"
if errorlevel 1 (
    echo ERROR: Patch failed
    exit /b 1
)
echo [OK] setup.py patched

REM --- Build and install stable-retro ---
echo.
echo [5/6] Building stable-retro (this takes a few minutes)...

REM Get cmake args from helper (zlib + python paths)
REM setup.py is already patched to use Ninja generator, so don't add -G here
for /f "tokens=*" %%i in ('python "%SCRIPT_DIR%setup_helper.py" cmake_args') do set CMAKE_ARGS=%%i

echo   CMAKE_ARGS=%CMAKE_ARGS%
echo.

pip install "%RETRO_BUILD_DIR%"
if errorlevel 1 (
    echo.
    echo ============================================================
    echo  ERROR: stable-retro build failed!
    echo ============================================================
    echo.
    echo Checklist:
    echo   [?] Are you in "x64 Native Tools Command Prompt for VS 2022"?
    echo   [?] Is conda env fzero activated?
    echo   [?] Is Windows Long Paths enabled in registry?
    echo   [?] Run: cl 2^>^&1 ^| findstr "x64"  -- must show x64
    echo.
    exit /b 1
)
echo [OK] stable-retro installed!

REM --- Install all other packages ---
echo.
echo [6/6] Installing remaining packages...

echo   Installing PyTorch with CUDA...
pip install torch --index-url https://download.pytorch.org/whl/cu124
if errorlevel 1 (
    echo   WARNING: CUDA torch failed, trying CPU version...
    pip install torch
)

echo   Installing RL and utility packages...
pip install "stable-baselines3[extra]>=2.7.0" "gymnasium>=1.0.0" "numpy>=1.26.0" "opencv-python>=4.8.0" "wandb>=0.16.0" "matplotlib>=3.8.0" "tensorboard>=2.15.0" "pytest>=7.0.0"

REM --- Verify ---
echo.
echo ============================================================
echo  Verifying installation...
echo ============================================================
python -c "import stable_retro; print('  stable-retro:', stable_retro.__version__)"
python -c "import stable_baselines3; print('  SB3:', stable_baselines3.__version__)"
python -c "import gymnasium; print('  gymnasium:', gymnasium.__version__)"
python -c "import torch; print('  torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import numpy; print('  numpy:', numpy.__version__)"
python -c "import cv2; print('  opencv:', cv2.__version__)"

echo.
echo ============================================================
echo  Setup complete!
echo ============================================================
echo.
echo Next steps:
echo   1. Place F-Zero ROM in: %PROJECT_DIR%\roms\
echo   2. Run tests: python -m pytest tests\ -v
echo   3. Train: python -m training.train --algo ppo
echo.

rmdir /s /q "%RETRO_BUILD_DIR%" 2>nul
endlocal
