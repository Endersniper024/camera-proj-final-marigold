source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd ~/autodl-tmp
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
python eval_diw_whdr.py \
  --diw_root /root/autodl-tmp/diw \
  --annot /root/autodl-tmp/diw/DIW_Annotations/DIW_test.csv \
  --checkpoint prs-eth/marigold-depth-v1-1 \
  --denoise_steps 1 --ensemble_size 1 --processing_res 768 \
  --limit 0 --log_every 2000 \
  --out_csv /root/autodl-tmp/diw_full.csv
echo DIW_FULL_DONE
