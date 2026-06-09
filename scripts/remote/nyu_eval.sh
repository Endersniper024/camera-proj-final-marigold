#!/bin/bash
set -e
echo "=== NYU EVAL START $(date) ==="
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY 2>/dev/null || true
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base

export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export BASE_DATA_DIR=/root/autodl-tmp/marigold_data
mkdir -p "$BASE_DATA_DIR/nyuv2"
cd /root/autodl-tmp/Marigold

CKPT=${1:-prs-eth/marigold-depth-v1-1}
NENS=${2:-10}
SUB=${3:-eval}

NYU_TAR="$BASE_DATA_DIR/nyuv2/nyu_labeled_extracted.tar"
NYU_EXPECTED=1085644800
ACTUAL=$(stat -c%s "$NYU_TAR" 2>/dev/null || echo 0)
if [ "$ACTUAL" != "$NYU_EXPECTED" ]; then
  echo "ERROR: NYU tar missing or wrong size ($ACTUAL != $NYU_EXPECTED)"; exit 1
fi
echo "NYU tar OK: $ACTUAL bytes"

echo "=== inference: ckpt=$CKPT ensemble=$NENS ==="
bash script/depth/eval/11_infer_nyu.sh "$CKPT" "$SUB" "$NENS"

echo "=== evaluate ==="
bash script/depth/eval/12_eval_nyu.sh "$SUB"

echo "=== METRICS ==="
cat output/${SUB}/nyu_test/eval_metric/*.txt 2>/dev/null || ls -R output/${SUB}/nyu_test/eval_metric
echo "=== NYU_EVAL_DONE $(date) ==="
