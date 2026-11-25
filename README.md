

> **[中文化] torchtitan**
>
> 此项目是 [pytorch/torchtitan](https://github.com/pytorch/torchtitan) 的中文翻译版本。
> - 原项目 Stars: 5975
> - 主语言: Python
> - 许可证: BSD-3-Clause
> - 翻译日期: 2026-11-25
> - 原始 README: [README_en.md](README_en.md)
>
> 说明：下方 README 正文为**原项目文档的中文翻译**，力求忠于原文；本页顶部的「中文快速上手」为**面向中文用户的补充导读**（非原文内容），目的是让你不用读完整篇 README 就能先把项目跑起来。
>
> 如有翻译不准确之处，欢迎提 Issue 或 PR。

---


## 🚀 中文快速上手（补充导读，非原文）

### 这是什么 / 解决什么问题

`torchtitan` 是 PyTorch 官方团队维护的一个 PyTorch 原生的大规模生成式 AI 模型训练平台，覆盖从预训练（pretraining）到监督微调（SFT）再到强化学习（RL）的完整流程。它把 FSDP2、张量并行（TP）、流水线并行（PP）、上下文并行（CP）、专家并行（EP）等并行方式组合成 N 维布局，而无需重写模型代码。项目定位是「干净、可读、可改」的研究型代码库，每次训练运行都是一个可读、可复制、可修改的 Python recipe（配置函数），而不是一堆命令行参数。支持 Llama 3、Qwen3、DeepSeek V3/V4、gpt-oss、Kimi、FLUX 等模型。

### 快速上手（最短路径）

1. 克隆仓库并进入目录：
   ```bash
   git clone https://github.com/pytorch/torchtitan && cd torchtitan
   ```
2. 安装 PyTorch nightly（把 `cu132` 换成你的 CUDA 或 ROCm 构建版本）：
   ```bash
   pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu132
   ```
3. 安装其余依赖：
   ```bash
   pip install -r requirements.txt
   ```
4. 跑一个 debug 模型（使用测试自带的 tokenizer，无需下载）：
   ```bash
   NGPU=1 ./run_train.sh   # Llama 3 debug model, 10 steps
   ```
5. 训练真实模型（需先下载 tokenizer，Llama 权重需在 Hugging Face 申请访问权限）：
   ```bash
   python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
   MODULE=torchtitan_recipes.models.llama3 CONFIG=llama3_8b ./run_train.sh   # 8 GPUs
   ```

### 环境要求

- **操作系统**：原文未明确说明支持哪些操作系统。
- **硬件**：需要 GPU；README 的安装命令以 CUDA/ROCm 构建为例，真实模型训练示例标注为 8 GPUs。AMD GPU 有官方优化分支（AMD-AGI/torchtitan-amd）。
- **Python 版本**：3.11+（README 与 pyproject.toml 的 `requires-python = ">=3.11"` 一致）。
- **PyTorch**：需使用 PyTorch nightly（`pip install --pre torch`），README 提示项目处于活跃开发中，要用最新特性请用较新的 nightly。
- **依赖**：`requirements.txt` 中包含 grain、datasets、pyarrow、tensorboard、wandb、tokenizers、safetensors、einops、pillow 等，其中 `attn-gym`、`torch_remat`、`torch_checkpointing` 直接从 GitHub 指定 commit 安装。
- **国内网络提示**：依赖中有多个包从 GitHub 直接拉取，PyTorch 也需从官方 nightly 源下载，国内环境可能需要配置镜像源或代理加速。

### 常见坑与注意事项

- **必须用 PyTorch nightly**：README 明确说明项目活跃开发中，最新特性依赖较新的 nightly 版本，用稳定版可能不兼容。
- **真实模型需要额外授权与下载**：Llama 权重需在 Hugging Face 申请访问权限，并需要 `--hf_token`；tokenizer 也需单独下载（debug 模型除外）。
- **真实训练示例按 8 GPU 给出**：`llama3_8b` 示例命令注释为 8 GPUs，单卡环境只能先跑 debug 模型验证流程。
- **依赖含 GitHub 直连的 commit 包**：`attn-gym`、`torch_remat`、`torch_checkpointing` 均从 GitHub 指定 commit 安装，网络受限时安装可能失败。

---



---

<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/images/logo/torchtitan-rectangle-torch-white.svg">
  <source media="(prefers-color-scheme: light)" srcset="assets/images/logo/torchtitan-rectangle-torch-black.svg">
  <img alt="torchtitan" src="assets/images/logo/torchtitan-rectangle-torch-black.svg" width="560">
</picture>

**一个 PyTorch 原生的规模化生成式 AI 模型预训练与后训练平台**

[![arXiv](https://img.shields.io/badge/arXiv-2410.06511-b31b1b.svg)](https://arxiv.org/abs/2410.06511)
[![ICLR](https://img.shields.io/badge/ICLR-2025-violet.svg)](https://iclr.cc/virtual/2025/poster/29620)
[![devlogs](https://img.shields.io/badge/PyTorch-DevLog-DE3412.svg)](https://docs.pytorch.org/devlogs/)
[![license](https://img.shields.io/badge/license-BSD_3--Clause-lightgrey.svg)](./LICENSE)
[![pip](https://img.shields.io/pypi/v/torchtitan?color=blue)](https://pypi.org/project/torchtitan/)
[![conda](https://img.shields.io/conda/vn/conda-forge/torchtitan?color=green)](https://anaconda.org/conda-forge/torchtitan)

[快速开始](#quick-start) · [模型](#supported-models) · [特性](#features) · [强化学习](torchtitan/rl/README.md) · [文档](#documentation) · [贡献](#contributing)

</div>

`torchtitan` 是一个净室实现的、PyTorch 原生的大规模训练方案。一份可读的代码库即可让模型从预训练走到监督微调，再到强化学习，组合 FSDP、张量并行、流水线并行、上下文并行和专家并行，无需重写模型。

- **可组合并行。** 模型自行声明如何分片；FSDP2、TP、PP、CP 和 EP 通过 `DeviceMesh` 和 DTensor 叠加成 N 维布局。
- **为研究而生。** 代码极简、易于修改；每次运行都是一个 Python [recipe](torchtitan/config/README.md)，可读、可复制、可修改。
- **一套技术栈覆盖预训练到强化学习。** [TitanRL](torchtitan/rl/README.md) 在训练器和 vLLM 生成中使用相同的模型定义与 kernel，并提供按位可复现的批次不变模式。
- **前沿模型与技术。** DeepSeek、Qwen、Kimi、gpt-oss、Llama 和 FLUX，支持 MXFP8/NVFP4 低精度、Dist-MoE、CUDA graphs 等。

> [!NOTE]
> `torchtitan` 正在积极开发中。要使用最新特性，请使用较新的 PyTorch nightly。

## 最新动态

- **[2026/10]** [DatologyAI](https://www.datologyai.com/) 的数据加载器 [Zephon](https://github.com/datologyai/zephon) 发布了其 [TorchTitan 集成](https://github.com/datologyai/torchtitan-zephon)。即使 GPU 数量变化，Zephon 也能保持相同的训练批次和顺序（弹性确定性）。参见 [pytorch.org 上的博客文章](https://pytorch.org/blog/bringing-elastically-deterministic-data-loading-to-torchtitan-with-zephon/)。
- **[2026/08]** [TitanRL](torchtitan/rl) 是一个易于修改的强化学习技术栈，用于扩展和调试。它在训练和 vLLM 生成中复用 TorchTitan 的模型定义与 kernel，并支持批次不变模式。
- **[2025/11]** AMD 发布了针对 AMD GPU 的 `torchtitan` [优化 Fork](https://github.com/AMD-AGI/torchtitan-amd/tree/main)。
- **[2025/10]** 我们发布了 `torchtitan` [v0.2.0](https://github.com/pytorch/torchtitan/releases)。
- **[2025/10]** SkyPilot 现已支持 `torchtitan`！参见[教程](https://docs.skypilot.co/en/latest/examples/training/torchtitan.html)。

<details>
<summary>更早的动态</summary>

- **[2025/07]** 我们发布了如何向 `torchtitan` 添加模型的[说明](torchtitan/models/README.md)。
- **[2025/04]** 我们的论文被 [ICLR 2025](https://iclr.cc/virtual/2025/poster/29620) 接收。
- **[2024/12]** GPU MODE 关于 torchtitan 的[讲座](https://www.youtube.com/watch?v=VYWRjcUqW6w)。
- **[2024/07]** PyTorch Conference 2024 上的[演讲](https://pytorch2024.sched.com/event/1fHn3)。

</details>

## 快速开始

**1. 安装**，使用 PyTorch nightly（Python 3.11+）。将 `cu132` 替换为你的 CUDA 或 ROCm 构建。

```bash
git clone https://github.com/pytorch/torchtitan && cd torchtitan
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu132
pip install -r requirements.txt
```

**2. 训练一个调试模型。** 它使用测试自带的 tokenizer，无需下载。

```bash
NGPU=1 ./run_train.sh   # Llama 3 调试模型，10 步
```

**3. 训练真实模型。** 下载 tokenizer（Llama 权重需要[在 Hugging Face 上申请访问权限](https://huggingface.co/meta-llama/Llama-3.1-8B)），然后按模块名和函数名选择 recipe：

```bash
python scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets tokenizer --hf_token=...
MODULE=torchtitan_recipes.models.llama3 CONFIG=llama3_8b ./run_train.sh   # 8 块 GPU
```

**4. 改成你自己的。** recipe 是一个返回完整配置的 Python 函数，因此修改一次运行只需写一个函数，而不用传一堆参数：

```python
# my_recipes.py
from torchtitan_recipes.models.llama3 import llama3_8b

def llama3_8b_tp2():
    config = llama3_8b()
    config.parallelism.tensor_parallel_degree = 2
    return config
```

```bash
MODULE=my_recipes CONFIG=llama3_8b_tp2 ./run_train.sh
```

完整模型说明见[配置指南](torchtitan/config/README.md)，已验证的 recipe 见 [`torchtitan_recipes`](torchtitan_recipes/README.md)。

## 支持的模型

| 模型 | 规模 | 模态 | 已验证 recipe | RL |
| --- | --- | --- | --- | :---: |
| [Llama 3](torchtitan/models/llama3) | 1B – 405B | 文本 | `llama3_8b`, `llama3_70b` | ✅ |
| [Qwen3](torchtitan/models/qwen3) | 0.6B – 32B 稠密；30B-A3B、235B-A22B MoE | 文本 | `qwen3_14b` | ✅ |
| [Qwen3.5](torchtitan/models/qwen3_5) / [3.6](torchtitan/models/qwen3_6) / [3.8](torchtitan/models/qwen3_8) | 0.8B – 397B-A17B；2.4T-A95B（3.8） | 文本、视觉 | | ✅ 文本 |
| [DeepSeek V3](torchtitan/models/deepseek_v3) | 16B、236B、671B | 文本 | `deepseek_v3_671b`（+ Dist-MoE、NVFP4） | |
| [DeepSeek V4](torchtitan/models/deepseek_v4) | Flash、Pro | 文本 | | 即将支持 |
| [gpt-oss](torchtitan/models/gpt_oss) | 20B、120B | 文本 | | ✅ |
| [Kimi K2.7](torchtitan/models/kimi_k2_7) | Moonlight-16B-A3B、Kimi-VL-A3B、K2.5 | 文本、视觉 | | |
| [Kimi K3](torchtitan/models/kimi_k3) | K3 | 文本、视觉 | | 即将支持 |
| [Muse Glimmer](torchtitan/models/muse_glimmer) | 30B | 文本、视觉 | `muse_glimmer_30b` | ✅ |
| [Muse-Spark-Open](torchtitan/models/muse_spark) | 816B-A42B | 文本、视觉、音频 | `muse_spark_816b_a42b_dist_moe_bf16_spmd_256` | ✅ 文本 |
| [FLUX](torchtitan/models/flux) | dev、schnell | 文生图 | `flux_dev`, `flux_schnell` | |

每个模型都有在 CI 中运行的调试配置。已验证的 recipe 已在训练、收敛或性能工作流中运行过。要添加模型，参见[模型指南](torchtitan/models/README.md)。

## 特性

<table>
<tr>
<td valign="top" width="50%">

**并行**
- [FSDP2](torchtitan/distributed/FSDP.md)、HSDP 和 DDP
- 张量并行，包括异步 TP
- [流水线并行](torchtitan/distributed/PIPELINE_PARALLEL.md)：1F1B、Interleaved 1F1B、GPipe 及其他 `torch.distributed.pipelining` 调度
- 上下文并行：all-gather KV 和 Ulysses
- 专家并行，支持 DeepEP / HybridEP 分发
- [Dist-MoE](torchtitan/models/common/dist_moe/README.md)：在 Blackwell 上融合分发、专家计算与合并

**内存与性能**
- [激活检查点](torchtitan/distributed/ACTIVATION_REMAT.md)：全量、选择性和区域
- [激活卸载](torchtitan/distributed/ACTIVATION_OFFLOAD.md)：将激活检查点保留的激活卸载到 CPU
- 区域 `torch.compile` 和 CUDA graphs
- [MXFP8](torchtitan/quantization/mxfp8/README.md) 和 [NVFP4](torchtitan/quantization/nvfp4/README.md) 训练
- FlashAttention 3/4、FlexAttention 和 varlen attention
- [BF16 优化器状态](torchtitan/components/optim/BF16_OPTIMIZER_STATES.md)和 FSDP CPU 卸载

**训练**
- [Grain 数据流水线](torchtitan/components/data/README.md)，支持打包、混合和可检查点状态
- 预训练、聊天数据集上的 SFT 和 LoRA
- Muon（DistMuon）、EMA、多 token 预测
- Warmup-stable-decay 学习率调度和[验证](docs/evaluation.md)

</td>
<td valign="top" width="50%">

**使用 [TitanRL](torchtitan/rl/README.md) 进行后训练**
- 训练器和 vLLM 共享同一份模型定义
- 批次不变模式下的[按位训练器/生成器一致性](torchtitan/rl/BITWISE_PARITY.md)
- 由 Monarch 编排的异步 rollout；通过 TorchStore 从训练器向生成器传输模型状态字典
- GRPO 和 DAPO 损失；[DAPO Math](torchtitan/rl/examples/dapo_math)、[Search-R1](torchtitan/rl/examples/search_r1) 和 [Verifiers](torchtitan/rl/examples/verifiers) 示例

**检查点**
- [分布式检查点](torchtitan/components/checkpointer/README.md)，包括异步保存
- 直接加载和保存 Hugging Face safetensors
- HF 与 DCP 之间的[转换脚本](scripts/checkpoint_conversion)

**调试与可观测性**
- [指标](torchtitan/observability/METRICS.md)：TensorBoard 或 W&B 中的损失、内存、吞吐量、TFLOPs 和 MFU
- [性能分析与内存快照](torchtitan/observability/README.md)
- [Flight Recorder 及其他调试工具](docs/debugging.md)
- 在单块 GPU 上对大规模布局进行假后端试运行
- 确定性模式和跨提交的[损失对比](scripts/loss_compare.py)
- [SDC 重放](torchtitan/observability/SILENT_DATA_CORRUPTION.md)和[结构化日志](torchtitan/observability/structured_logger/README.md)

**可扩展性**
- 通过 `--module` 和 `--config` 选择 Python [recipe](torchtitan/config/README.md)
- 用于自定义 kernel 和模块的[组件覆盖](torchtitan/config/OVERRIDE.md)
- [实验](torchtitan/experiments)：GraphTrainer、TorchFT、Transformers 建模后端

</td>
</tr>
</table>

我们报告了多达 512 块 GPU 上的[性能](docs/benchmarks/llama3_h100_202412_torchtitan.md)，并验证了跨各种并行方式与技术的[损失收敛](torchtitan/models/llama3/README.md#loss-convergence)。

## 测试状态

| 硬件 | 集成测试 | 单元测试 |
| --- | --- | --- |
| CPU | | [![CPU Unit Test](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_cpu.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_cpu.yaml?query=branch%3Amain) |
| NVIDIA GPU | [![Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test.yaml?query=branch%3Amain) [![H100 Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_h100.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_h100.yaml?query=branch%3Amain) [![B200 Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_b200.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_b200.yaml?query=branch%3Amain) | [![GPU Unit Tests](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_gpu.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_gpu.yaml?query=branch%3Amain) |
| AMD GPU (ROCm) | [![Integration Tests (ROCm)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_rocm.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_rocm.yaml?query=branch%3Amain) | |
| RL (NVIDIA GPU) | [![RL Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_rl.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_rl.yaml?query=branch%3Amain) [![RL B200 Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_rl_b200.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_rl_b200.yaml?query=branch%3Amain) | [![RL GPU Unit Tests](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_gpu_rl.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/unit_test_gpu_rl.yaml?query=branch%3Amain) |

## 安装

[快速开始](#quick-start)从源码安装，这是获取最新特性的推荐方式。其他选项：

<details>
<summary><b>Nightly 构建</b></summary>

```bash
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu132 --force-reinstall
pip install --pre torchtitan --index-url https://download.pytorch.org/whl/nightly/cu132
```

将 `cu132` 替换为其他 CUDA 版本或 ROCm 构建（例如 `rocm10.0`）。

</details>

<details>
<summary><b>稳定版本</b></summary>

```bash
pip install torchtitan
# 或
conda install conda-forge::torchtitan
```

每个稳定版本都固定了其验证时使用的 `torch` 和 `torchao` nightly 版本；参见 [RELEASE.md](RELEASE.md)。

</details>

<details>
<summary><b>可选依赖</b></summary>

- **低精度（MXFP8、NVFP4）：** 安装 `torchao` nightly CI 固定的版本，使用与你的 `torch`