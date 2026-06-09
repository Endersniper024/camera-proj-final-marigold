# 环境与依赖说明

本项目在「本地 + GPU 服务器」两套环境下复现 Marigold：本地用于轻量定性结果与代码调试，重量级评测（NYUv2、DIW、ETH3D、消融）在 GPU 服务器上完成。

## 关键依赖

- Python 3.10。
- 与官方 pipeline 兼容的钉死版本：`diffusers==0.32.2`、`transformers==4.46.3`，以及匹配本机 CUDA 的 PyTorch GPU 版本。
- 远程环境：较新架构的 NVIDIA GPU 需要 CUDA 12.8 / PyTorch `cu128` 轮子，完整依赖见 `remote-cu128.txt`。

## 复现要点

- diffusers / transformers / torch 版本必须锁定，否则官方 `MarigoldPipeline` 可能不兼容。
- 权重从 HuggingFace 下载；xet 后端在部分权重上会超时，需设 `HF_HUB_DISABLE_XET=1` 或切换镜像 `HF_ENDPOINT=https://hf-mirror.com`。
- 官方评测脚本中的 `metric.py` 在较新 pandas 下需小幅修补（dtype 与只读数组行为）。
- ETH3D 数据量很大，服务器直连下载慢且占磁盘，最终采用本地下载、远程运行的方式。
- Windows 下注意行尾（CRLF）、符号链接权限与路径编码。

## 安装

远程环境的安装步骤见 `../scripts/remote/setup.sh`。
