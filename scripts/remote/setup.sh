#!/bin/bash
set -e
echo "=== START $(date) ==="
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
cd /root/autodl-tmp

export HF_HOME=/root/autodl-tmp/hf_cache
mkdir -p "$HF_HOME"

echo "=== clone Marigold (robust) ==="
git config --global http.version HTTP/1.1
git config --global http.postBuffer 524288000
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999
if [ ! -d Marigold/.git ]; then
  rm -rf Marigold
  for attempt in 1 2 3 4 5; do
    echo "clone attempt $attempt ..."
    if git clone --depth 1 https://github.com/prs-eth/Marigold.git; then
      echo "clone OK"; break
    fi
    echo "clone failed, retrying"; rm -rf Marigold; sleep 3
  done
else
  echo "Marigold already cloned"
fi
test -d Marigold/.git || { echo "ERROR: clone failed after retries"; exit 1; }

echo "=== create conda env (py3.10) ==="
source /root/miniconda3/etc/profile.d/conda.sh
if ! conda env list | grep -q "^marigold "; then
  conda create -n marigold python=3.10 -y
fi
conda activate marigold

echo "=== install torch 2.8.0 + cu128 (Blackwell/RTX5090) ==="
pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128

echo "=== install marigold deps ==="
pip install "diffusers==0.32.2" "transformers==4.46.3" accelerate matplotlib scipy "huggingface_hub==0.25.2" tabulate

echo "=== verify ==="
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'dev', torch.cuda.get_device_name(0))"
python -c "import diffusers, transformers; print('diffusers', diffusers.__version__, 'transformers', transformers.__version__)"
cd /root/autodl-tmp/Marigold
python -c "import sys; sys.path.insert(0,'.'); from marigold import MarigoldDepthPipeline; print('marigold import OK')"
echo "=== SETUP_DONE $(date) ==="
