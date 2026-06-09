#!/usr/bin/env bash
set -e
set -x
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/Marigold
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
IN=/root/autodl-tmp/other_data_prep
OUT=/root/autodl-tmp/other_data_out
mkdir -p "$OUT"

echo "=== DEPTH (v1-1, 1 step, ens10, native res) ==="
python script/depth/run.py --checkpoint prs-eth/marigold-depth-v1-1 \
  --input_rgb_dir "$IN" --output_dir "$OUT/depth" \
  --denoise_steps 1 --ensemble_size 10 --processing_res 0

echo "=== NORMALS (v1-1, 1 step, ens10, native res) ==="
python script/normals/run.py --checkpoint prs-eth/marigold-normals-v1-1 \
  --input_rgb_dir "$IN" --output_dir "$OUT/normals" \
  --denoise_steps 1 --ensemble_size 10 --processing_res 0

echo "=== IID appearance (v1-1, ens1, native res) ==="
python script/iid/run.py --checkpoint prs-eth/marigold-iid-appearance-v1-1 \
  --input_rgb_dir "$IN" --output_dir "$OUT/iid_appearance" \
  --ensemble_size 1 --processing_res 0

echo "ALL_OTHERDATA_DONE"
