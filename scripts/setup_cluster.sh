#!/bin/bash
# ============================================================
# F-Zero RL — Cluster Setup Script
#
# Usage: ssh deepx-4 "bash /home/kevinzyz/yincheng/fzero/scripts/setup_cluster.sh"
# ============================================================
set -e

PROJECT_DIR="/home/kevinzyz/yincheng/fzero"
cd "$PROJECT_DIR"

echo "============================================================"
echo " F-Zero RL Cluster Setup"
echo "============================================================"
echo

# --- Create venv ---
VENV_DIR="$PROJECT_DIR/.venv"
if [ ! -d "$VENV_DIR" ]; then
    echo "[1/4] Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"
echo "[OK] venv: $VENV_DIR"
echo "[OK] Python: $(python --version)"

# --- Upgrade pip ---
echo
echo "[2/4] Upgrading pip..."
pip install --upgrade pip -q

# --- Install all deps ---
echo
echo "[3/4] Installing dependencies..."
pip install -r requirements.txt -q

echo
echo "[4/4] Installing PyTorch with CUDA..."
pip install torch --index-url https://download.pytorch.org/whl/cu124 -q

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
echo "  CPU cores: $(nproc)"

echo
echo "============================================================"
echo " Setup complete!"
echo "============================================================"
echo
echo "To train:"
echo "  source $VENV_DIR/bin/activate"
echo "  cd $PROJECT_DIR"
echo "  python -m training.train --algo ppo --n-envs 80"
echo
