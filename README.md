# Camera Final Assignment - Marigold Reproduction

This repository-style folder contains the code and evidence used for the final report section on Marigold depth estimation reproduction.

## Structure

- `data/`
  - `processed/my_data/`: preprocessed self-captured images.
  - `processed/other_data/`: preprocessed teammate/web challenge images.
  - `annotations/`: category manifest and manually annotated depth pairs.
- `scripts/`
  - local data preparation, result collation, visualization, comparison, and the minimal Marigold reimplementation.
  - `scripts/remote/`: server-side scripts for NYUv2, DIW, ETH3D, ablation, and multi-task Marigold runs.
- `results/`
  - selected visualizations, CSV/JSON/TXT metrics, montages, and result README files.
  - Large `.npy` tensors are intentionally excluded.
- `requirements/`
  - environment notes for local and remote reproduction.
