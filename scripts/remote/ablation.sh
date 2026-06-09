#!/bin/bash
set -e
echo "=== ABLATION START $(date) ==="
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_ENDPOINT=https://hf-mirror.com
export BASE_DATA_DIR=/root/autodl-tmp/marigold_data
cd /root/autodl-tmp/Marigold

FULL=data_split/nyu_depth/labeled/filename_list_test.txt
SUB=data_split/nyu_depth/labeled/filename_list_test_sub.txt
awk 'NR%6==1' "$FULL" > "$SUB"
N=$(wc -l < "$SUB")
echo "subset size = $N"

CFG=config/dataset_depth/data_nyu_test_sub.yaml
cp config/dataset_depth/data_nyu_test.yaml "$CFG"
sed -i 's#filename_list_test.txt#filename_list_test_sub.txt#' "$CFG"

SUMMARY=/root/autodl-tmp/ablation_summary.csv
echo "tag,steps,ensemble,abs_rel,delta1,rmse,sec_per_img" > "$SUMMARY"

run_cfg () {
  local steps=$1; local ens=$2; local tag="s${steps}_e${ens}"
  local odir=output/ablation/$tag
  rm -rf "$odir"; mkdir -p "$odir"
  echo "--- running $tag (steps=$steps ens=$ens) ---"
  local t0=$(date +%s)
  python script/depth/infer.py --checkpoint prs-eth/marigold-depth-v1-1 --seed 1234 \
    --base_data_dir "$BASE_DATA_DIR" --denoise_steps "$steps" --ensemble_size "$ens" \
    --processing_res 0 --dataset_config "$CFG" --output_dir "$odir/prediction" \
    > "$odir/infer.log" 2>&1
  local t1=$(date +%s)
  python script/depth/eval.py --base_data_dir "$BASE_DATA_DIR" \
    --dataset_config "$CFG" --alignment least_square \
    --prediction_dir "$odir/prediction" --output_dir "$odir/eval_metric" \
    > "$odir/eval.log" 2>&1
  python - "$odir/eval_metric/per_sample_metrics.csv" "$tag" "$steps" "$ens" "$t0" "$t1" "$N" "$SUMMARY" <<'PY'
import sys, csv
csvf, tag, steps, ens, t0, t1, N, summary = sys.argv[1:9]
rows=list(csv.DictReader(open(csvf)))
def avg(k):
    vals=[float(r[k]) for r in rows]; return sum(vals)/len(vals)
ar=avg('abs_relative_difference'); d1=avg('delta1_acc'); rmse=avg('rmse_linear')
spi=(int(t1)-int(t0))/int(N)
with open(summary,'a') as f:
    f.write(f"{tag},{steps},{ens},{ar:.4f},{d1:.4f},{rmse:.4f},{spi:.2f}\n")
print(f"{tag}: AbsRel={ar:.4f} d1={d1:.4f} rmse={rmse:.4f} {spi:.2f}s/img")
PY
}

run_cfg 1 1
run_cfg 4 1
run_cfg 10 1
run_cfg 50 1
run_cfg 1 5
run_cfg 1 10

echo "=== SUMMARY ==="
cat "$SUMMARY"
echo "=== ABLATION_DONE $(date) ==="
