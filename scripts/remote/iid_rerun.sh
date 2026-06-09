source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
pkill -f 'script/iid/run.py' 2>/dev/null
pkill -f 'other_data_run.sh' 2>/dev/null
sleep 2
cd /root/autodl-tmp/Marigold
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/hf_cache
export HF_HUB_DISABLE_XET=1
export HF_XET_DISABLE=1
python script/iid/run.py --checkpoint prs-eth/marigold-iid-appearance-v1-1 \
  --input_rgb_dir /root/autodl-tmp/other_data_prep \
  --output_dir /root/autodl-tmp/other_data_out/iid_appearance \
  --ensemble_size 1 --processing_res 0
echo IID_RERUN_DONE
