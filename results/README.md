# results/ — Marigold 实验结果

本目录的组织方式与组员的 Depth-Anything 输出（`da2-*` / `da3-*`）保持一致，便于并排对比。组员的输出与对照在其各自的提交中，不随本包附带。

## 目录一览

| 目录 | 图像集 | 张数 |
|---|---|---|
| `marigold-other/` | my_data（自拍，`NN__name`） | 19 |
| `marigold-phone_photo/` | other_data 手机照片（`IMG_*`） | 13 |
| `marigold-web/` | other_data 网图难例（`webNNN`） | 7 |
| `marigold-normals-*` | 对应图像集的表面法线（vis） | 17 / 13 / 7 |
| `marigold-iid-*` | 对应图像集的本征 albedo + material（vis） | 17×2 / 13×2 / 7×2 |
| `marigold-reimpl/` | 从零手写推理样例（与官方一致） | 1 |
| `montages/` | RGB｜深度｜法线｜albedo｜material 并排大图 + `by_category/` 合集 | 20 + 8 |
| `nyu/eval_full/` | NYUv2 定量（AbsRel 0.0580 / δ1 0.9610） | 654 |
| `diw/eval_full/` | DIW 全量 WHDR（0.1318） | 74441 对 |
| `diw/eval_sub500/` | DIW 500 子集 sanity（WHDR 0.1440） | 500 |
| `eth3d/eval_full/` | ETH3D 稠密深度（AbsRel 6.93% / δ1 95.67%） | 454 |
| `other_data/` | web 难例自定义 WHDR 逐点对结果 | 35 对 |

> `marigold-normals-other` / `marigold-iid-other` 为 my_data 的 17 张，不含后加的 2 张玻璃图（那 2 张当时只跑了深度）。
> 体积较大的 `.npy` 张量未随包附带，仅保留可视化与指标文件。

## 每个图像集目录的内容

- `<name>.png` — Marigold 原生 Spectral 伪彩深度。
- `recolor/<name>.png` — 统一用 `Spectral_r` 重新着色（近亮远暗，逐图 2–98 分位归一），便于跨方法公平对比。
- `summary.json` — 推理设置（checkpoint=marigold-depth-v1-1，denoise_steps=1，ensemble_size=10，原生分辨率等）。
- `runtime.csv` — 逐图 宽/高/深度 min-max 等；`elapsed_ms` 留空（本批为整批跑，未逐图计时；速度见报告，GPU 上约 0.12–0.23 s/张）。

> 深度方向约定：Marigold 原始深度**值越大越远**（与 Depth-Anything 的"越大越近"相反），跨方法可视化时已统一重着色。

## DIW eval_full（`diw/eval_full/`）

- `summary.json` / `model_summary.csv` — 模型级汇总：**Marigold WHDR=0.1318（13.18%）**，selected_direction=larger_farther，total_pairs=74441，correct=64633。
- `pair_results.csv` — 逐点对结果：`model,image,pair_id,ax,ay,bx,by,closer,prediction,correct,depth_a,depth_b,...`。
  - `closer` 由 DIW 关系换算（`>`→B 近，`<`→A 近）；`prediction`=深度较小者为近。

> 参考（同测试集下组员结果，详见组员提交）：da2_vitl WHDR≈0.111、da3mono≈0.114；三者均在 in-the-wild 相对深度上达到 ~11–13% 的水平。

## 生成脚本

`../scripts/build_output.py` 将 Marigold 的深度 / 法线 / 本征分解结果统一整理到上述可比较的目录结构；可视化大图由 `../scripts/make_montages.py` 生成。
