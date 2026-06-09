#!/bin/bash
set -e
echo "=== LCM EVAL START $(date) ==="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export BASE_DATA_DIR=/root/autodl-tmp/marigold_data
cd /root/autodl-tmp/Marigold

CFG=config/dataset_depth/data_nyu_test_sub.yaml
SUMMARY=/root/autodl-tmp/lcm_summary.csv
N=$(wc -l < data_split/nyu_depth/labeled/filename_list_test_sub.txt)
echo "tag,steps,ensemble,abs_rel,delta1,rmse,sec_per_img" > "$SUMMARY"

run_cfg () {
  local steps=$1; local ens=$2; local tag="lcm_s${steps}_e${ens}"
  local odir=output/ablation/$tag
  rm -rf "$odir"; mkdir -p "$odir"
  echo "--- $tag ---"
  local t0=$(date +%s)
  python script/depth/infer.py --checkpoint prs-eth/marigold-depth-lcm-v1-0 --seed 1234 \
    --base_data_dir "$BASE_DATA_DIR" --denoise_steps "$steps" --ensemble_size "$ens" \
    --processing_res 0 --dataset_config "$CFG" --output_dir "$odir/prediction" \
    > "$odir/infer.log" 2>&1
  local t1=$(date +%s)
  python script/depth/eval.py --base_data_dir "$BASE_DATA_DIR" --dataset_config "$CFG" \
    --alignment least_square --prediction_dir "$odir/prediction" \
    --output_dir "$odir/eval_metric" > "$odir/eval.log" 2>&1
  python - "$odir/eval_metric/per_sample_metrics.csv" "$tag" "$steps" "$ens" "$t0" "$t1" "$N" "$SUMMARY" <<'PY'
import sys, csv
csvf, tag, steps, ens, t0, t1, N, summary = sys.argv[1:9]
rows=list(csv.DictReader(open(csvf)))
avg=lambda k: sum(float(r[k]) for r in rows)/len(rows)
with open(summary,'a') as f:
    f.write(f"{tag},{steps},{ens},{avg('abs_relative_difference'):.4f},{avg('delta1_acc'):.4f},{avg('rmse_linear'):.4f},{(int(t1)-int(t0))/int(N):.2f}\n")
print(f"{tag}: AbsRel={avg('abs_relative_difference'):.4f} d1={avg('delta1_acc'):.4f}")
PY
}

run_cfg 1 1
run_cfg 4 1
run_cfg 1 10

echo "=== LCM SUMMARY ==="; cat "$SUMMARY"
echo "=== LCM_DONE $(date) ==="
