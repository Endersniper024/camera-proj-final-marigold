# Marigold 复现实验说明

本文档说明 Marigold 单目深度估计的复现工作，是在课程 Pre 基础上补充的完整实验部分。工作包括：运行官方代码完成深度估计，并扩展到法线与本征分解；在 NYUv2、DIW、ETH3D 三个公开基准上做定量复现；对去噪步数、集成数与加速版本做消融；在自拍图与难例数据上做定性分析；以及从零手写一份推理实现以验证对原理的理解。文中每节末尾给出对应的结果与脚本文件，便于在本目录中查阅。

## 1. 定量复现与评测

### NYUv2 全量定量复现

在 NYUv2 test split 全部 654 张图上完成评测，使用 `marigold-depth-v1-1`，`denoise_steps=1`，`ensemble_size=10`，原生分辨率，按官方协议做 least-square 对齐。

| 指标 | 本次复现 |
|---|---:|
| AbsRel ↓ | 0.0580 |
| δ1 ↑ | 0.9610 |
| RMSE ↓ | 0.2279 |
| log10 ↓ | 0.0248 |

复现数值与论文报告的 NYUv2 结果基本一致，表明官方 checkpoint 与本项目的评测流程可信。

结果文件：`results/nyu/eval_full/` 下的 `nyu_v1_1_metrics.txt`、`summary.json`、`nyu_v1_1_per_sample.csv`。

### DIW 全量相对深度评测

在 DIW full test 上评测，共 74,441 对相对深度点。DIW 只判断两个像素谁更近，指标为 WHDR（越低越好）。Marigold 输出仿射不变深度，WHDR 不依赖绝对尺度，可直接按深度大小比较。

| 指标 | 本次复现 |
|---|---:|
| WHDR ↓ | 0.1318 / 13.18% |
| total pairs | 74,441 |
| missing | 0 |
| correct | 64,633 |

Marigold 在 in-the-wild 相对深度排序上零样本表现很强，优于早期 DIW 论文中的有监督方法。

结果文件：`results/diw/eval_full/summary.json`、`model_summary.csv`；评测脚本 `scripts/remote/eval_diw_whdr.py`。

### ETH3D 户外高分辨率评测

为对齐论文中的户外基准，补跑了 ETH3D dense depth benchmark，沿用官方脚本协议：`processing_res=756`，`ensemble=10`，`denoise_steps=1`，least-square 对齐。

| 指标 | 本次复现 |
|---|---:|
| AbsRel ↓ | 0.0693 / 6.93% |
| δ1 ↑ | 0.9567 / 95.67% |

ETH3D 结果接近论文值，说明 Marigold 在户外高分辨率场景下同样具备良好的零样本泛化能力。

结果文件：`results/eth3d/eval_full/eval_metrics-least_square.txt`、`summary.json`；运行脚本 `scripts/remote/eth3d_run.sh`。

### 消融：去噪步数与集成数

在 NYUv2 的 109 张子集上对去噪步数和 ensemble size 做消融。

| 配置 | 步数 | 集成 | AbsRel ↓ | δ1 ↑ | 耗时/张 |
|---|---:|---:|---:|---:|---:|
| s1_e1 | 1 | 1 | 0.0588 | 0.9605 | 0.23s |
| s4_e1 | 4 | 1 | 0.0597 | 0.9612 | 0.34s |
| s10_e1 | 10 | 1 | 0.0606 | 0.9609 | 0.62s |
| s50_e1 | 50 | 1 | 0.0614 | 0.9600 | 1.92s |
| s1_e5 | 1 | 5 | 0.0582 | 0.9620 | 1.54s |
| s1_e10 | 1 | 10 | 0.0582 | 0.9617 | 5.86s |

对 v1-1 而言，单步推理已足够强，增加步数没有带来提升，反而略慢且略差；集成数从 1 增到 5 有小幅提升，5 到 10 收益已很小但耗时明显上升。综合精度与速度，`denoise_steps=1` 搭配 `ensemble_size=5` 或 `10` 是较合理的设置。

结果文件：`results/nyu/eval_full/ablation_summary.csv`；运行脚本 `scripts/remote/ablation.sh`。

### LCM 加速版对比

对比了 Marigold v1-1 与 LCM v1-0。LCM 这一早期加速版精度略低于 v1-1，需要更多步数或更大 ensemble 才能接近。v1-1 是更成熟的单步蒸馏版本，在速度与精度的平衡上更适合作为复现主模型。

结果文件：`results/nyu/eval_full/lcm_summary.csv`；运行脚本 `scripts/remote/lcm_eval.sh`。

### 真实难例分析

在 web 难例上手工标注了 35 对相对深度点并计算 WHDR，整体 WHDR 为 20.0%。失败主要集中在镜面反射与强迫透视场景——镜子会被模型当成真实空间的延伸，手捏埃菲尔铁塔这类强迫透视会扰乱尺度判断。可见 Marigold 的扩散先验能给出视觉上合理的深度，但对反射、折射和尺度错觉仍缺乏真正的几何理解。

数据与结果：`data/annotations/web_depth_pairs.csv`、`results/other_data/web_whdr_per_pair.csv`；评测脚本 `scripts/remote/web_whdr.py`。

### 多任务扩展

除深度外，还跑通了官方 normals 与 intrinsic image decomposition 输出，在自有图片与网图难例上生成了深度、法线、albedo/material 的并排结果，说明 Marigold 框架的扩散先验不止用于单目深度，也可迁移到表面法线与本征分解。需要说明的是，表面法线与本征分解并不在 CVPR 2024 正文范围内，而是作者后续放出的官方 checkpoint 能力；本项目把它们在自有数据上一并跑通并做定性验证，作为相对 Pre 的扩展。

结果文件：`results/montages/`、`results/marigold-normals-*`、`results/marigold-iid-*`。

## 2. 实验细节

### 数据与预处理

自拍图片先经 EXIF 方向修正、RGB 转换、长边缩放到 1536，再统一命名；网图难例数据整理为 8 类（室内博物馆玻璃、室外街景、水景山景、低光夜景、反射折射、强迫透视、合成错觉等）。所有深度结果统一整理成与 Depth Anything 输出相近的 schema，便于横向比较。

相关脚本：`scripts/process_my_data.py`、`scripts/classify_other_data.py`、`scripts/build_output.py`。

### 评测协议

- NYUv2 与 ETH3D 使用 least-square 对齐，因为 Marigold 输出的是仿射不变深度。
- DIW 与 web 难例只比较两点远近，无需绝对尺度对齐。
- 对比可视化时需注意深度方向：Marigold 原始值越大越远，部分 Depth Anything 输出约定相反，因此展示时统一重着色。

### 运行环境

轻量定性结果与代码调试在本地完成；NYUv2、DIW、ETH3D 与消融等重量级评测在 GPU 服务器上完成。依赖版本（torch / diffusers / transformers 等）需锁定，否则官方 pipeline 可能不兼容。环境说明见 `requirements/README.md`，安装脚本见 `scripts/remote/setup.sh`。

### 复现中的工程问题

- diffusers / transformers / torch 版本需锁定，否则官方 pipeline 可能不兼容。
- HuggingFace xet 后端在部分权重下载时会超时，需要禁用或切换镜像。
- ETH3D 数据量很大，服务器直连下载慢且占用磁盘，最终采用本地下载、远程运行的方式。
- pandas 版本变化会影响官方评测脚本中的 dtype 与只读数组行为，需要修补。
- Windows 下的行尾、符号链接与路径编码也会影响脚本迁移。

## 3. 代码工作

本部分的代码工作包括：

- 自有数据预处理脚本，完成 EXIF 修正、缩放、重命名与输入整理。
- 网图数据的分类与 manifest 生成脚本。
- 输出整理脚本，将 Marigold 的深度、法线、本征分解结果统一到可比较的目录结构。
- DIW WHDR 与 web 难例 WHDR 的评测脚本。
- NYUv2 消融、ETH3D、LCM 对比等远程运行脚本。
- montage 生成脚本，用于多任务定性图。
- 手写的 `marigold_minimal.py`，复现 Marigold 推理核心流程。

## 4. 手写推理实现

`scripts/marigold_minimal.py` 没有直接调用官方 `run.py`，而是用 diffusers 的基础组件手动实现了主要推理流程：

1. 用 Stable Diffusion VAE 编码 RGB 图像，得到条件 latent。
2. 从随机噪声初始化深度 latent。
3. 每个 DDIM 步骤中拼接 RGB latent 与深度 latent，送入 UNet 预测噪声。
4. 用 scheduler 更新深度 latent。
5. 通过 VAE decoder 解码出深度图。
6. 对多次预测做 scale-shift 对齐并取中位数集成。
7. 输出仿射不变深度与可视化结果。

该实现重写了核心推理路径，并与官方输出对齐验证，用于确认对模型原理的理解，而非仅调用现成脚本。

## 5. AI 使用说明

本项目中 AI 主要用于辅助整理复现计划、定位依赖与环境问题、以及辅助汇总指标。最终实验结论均基于实际运行得到的本地或远程输出文件，包括 NYUv2、DIW、ETH3D 指标文件与可视化结果。
