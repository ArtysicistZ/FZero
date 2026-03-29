#!/bin/bash
# ============================================================
# F-Zero RL — WSL Environment Setup
#
# Run from WSL Ubuntu:
#   cd /mnt/d/Code/AI\ Project/RL/RL-Gaming
#   bash scripts/setup_wsl.sh
# ============================================================
set -e

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
echo "============================================================"
echo " F-Zero RL Environment Setup (WSL)"
echo " Project: $PROJECT_DIR"
echo "============================================================"
echo

# --- Check Python ---
PYTHON=python3.12
if ! command -v $PYTHON &>/dev/null; then
    echo "Installing Python 3.12..."
    sudo apt update && sudo apt install -y python3.12 python3.12-venv python3.12-dev
fi
echo "[OK] $($PYTHON --version)"

# --- Create venv in Linux filesystem (not /mnt/d/ — venv is slow/broken on NTFS) ---
VENV_DIR="$HOME/.venvs/fzero"
if [ ! -d "$VENV_DIR" ]; then
    echo
    echo "[1/4] Creating virtual environment at $VENV_DIR ..."
    mkdir -p "$HOME/.venvs"
    $PYTHON -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "[OK] venv activated: $VENV_DIR"

# --- Install pip essentials ---
echo
echo "[2/4] Upgrading pip..."
pip install --upgrade pip -q

# --- Install stable-retro (builds from source on Linux — works cleanly) ---
echo
echo "[3/4] Installing stable-retro + all dependencies..."
pip install stable-retro -q
pip install "stable-baselines3[extra]>=2.7.0" -q
pip install gymnasium numpy opencv-python-headless wandb matplotlib tensorboard pytest -q

# --- Install PyTorch with CUDA ---
echo
echo "[4/4] Installing PyTorch with CUDA..."
# WSL2 supports CUDA passthrough from Windows GPU drivers
pip install torch --index-url https://download.pytorch.org/whl/cu124 -q
if [ $? -ne 0 ]; then
    echo "  CUDA torch failed, installing CPU version..."
    pip install torch -q
fi

# --- Verify ---
echo
echo "============================================================"
echo " Verifying installation..."
echo "============================================================"
python -c "import stable_retro; print('  stable-retro:', stable_retro.__version__)"
python -c "import stable_baselines3; print('  SB3:', stable_baselines3.__version__)"
python -c "import gymnasium; print('  gymnasium:', gymnasium.__version__)"
python -c "import torch; print('  torch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -c "import numpy; print('  numpy:', numpy.__version__)"
python -c "import cv2; print('  opencv:', cv2.__version__)"

echo
echo "============================================================"
echo " Setup complete!"
echo "============================================================"
echo
echo "To activate the environment in future sessions:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd /mnt/d/Code/AI\ Project/RL/RL-Gaming"
echo
echo "Next steps:"
echo "  1. Place F-Zero ROM in: $PROJECT_DIR/roms/"
echo "  2. Run tests: python -m pytest tests/ -v"
echo "  3. Train: python -m training.train --algo ppo"
echo
