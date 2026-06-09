source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd ~/autodl-tmp/diw
echo '=== extracting DIW_test.tar.gz ==='
tar xzf DIW_test.tar.gz
echo "images extracted: $(ls DIW_test 2>/dev/null | wc -l)"
echo '=== free disk ==='
df -h ~/autodl-tmp | tail -1
echo '=== run 500-image sanity subset ==='
cd ~/autodl-tmp
python eval_diw_whdr.py \
  --diw_root /root/autodl-tmp/diw \
  --annot /root/autodl-tmp/diw/DIW_Annotations/DIW_test.csv \
  --checkpoint prs-eth/marigold-depth-v1-1 \
  --denoise_steps 1 --ensemble_size 1 --processing_res 768 \
  --limit 500 --shuffle_seed 0 --log_every 100 \
  --out_csv /root/autodl-tmp/diw_sub500.csv
