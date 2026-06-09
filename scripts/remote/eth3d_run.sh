#!/usr/bin/env bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_DISABLE_XET=1
export BASE_DATA_DIR=/root/autodl-tmp
export PYTHONIOENCODING=utf-8
cd /root/autodl-tmp/Marigold

echo "===== ETH3D INFER start $(date) ====="
bash script/depth/eval/31_infer_eth3d.sh prs-eth/marigold-depth-v1-1 eval 10
echo "===== ETH3D EVAL start $(date) ====="
bash script/depth/eval/32_eval_eth3d.sh eval
echo "===== DONE $(date) ====="
echo "--- metrics ---"
cat output/eval/eth3d/eval_metric/*.txt 2>/dev/null
echo "ETH3D_RUN OK"
