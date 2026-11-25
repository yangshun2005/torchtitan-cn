# Kimi K3

Kimi K3 将混合 Kimi Delta Attention（KDA）与 Multi-head Latent Attention（MLA）解码器，与 LatentMoE 及 MoonViT-V2 视觉编码器相结合。

## 前置条件

安装额外依赖：

```bash
pip install -r .ci/docker/requirements-vlm.txt
```

KDA 使用 Attention Gym 内核，需要 CUDA 计算能力 9.0 或更新的 NVIDIA GPU。Attention Gym 在 SM100/SM103 上运行其 CuTe 内核，在其他 GPU 上运行其 Triton 内核。

## 架构

Kimi K3 基于 Kimi Delta Attention（KDA）和 Attention Residuals（AttnRes）构建，包含 69 个 KDA 层和 24 个 Gated MLA 层。Stable LatentMoE 为每个 token 从 896 个专家中选择 16 个，MoonViT-V2 提供原生视觉输入。

`Kimi-K3` 和 `debugmodel` 两种规格仅支持文本：它们构建 `KimiK3Model`（不含视觉编码器的解码器），并从多模态检查点读取语言模型。`_mm` 规格构建 `KimiK3MultimodalModel`，额外加入 MoonViT-V2。

## 已发布模型配置

以下数值遵循
[官方 Kimi K3 模型卡](https://huggingface.co/moonshotai/Kimi-K3)，描述已发布的模型。

| 组件 | 配置 |
|-----------|---------------|
| 架构 | 混合专家（MoE） |
| 参数量 | 总计 2.8T，激活 104B |
| 解码器 | 93 层，1 个稠密层，隐藏维度 7168，96 个注意力头 |
| 注意力 | 69 个 KDA 层和 24 个 Gated MLA 层，上下文长度 1048576 |
| LatentMoE | 维度 3584，每个专家隐藏维度 3072，896 个专家，top-16 路由，2 个共享专家 |
| 词表 | 160K |
| 激活函数 | SiTU-GLU |
| 视觉编码器 | MoonViT-V2，401M 参数 |
| 量化 | MXFP4 权重和 MXFP8 激活，采用量化感知训练 |
| 模态 | 文本和图像 |

## 支持的并行方式

| 特性 | 说明 |
|---------|-------|
| FSDP2 / HSDP | 解码器按层分片；视觉编码器作为独立单元分片 |
| 上下文并行 | KDA 使用 Attention Gym 原生 CP；MLA 支持 all-gather K/V（连续或头尾分区）以及 Ulysses（连续分区） |
| 张量 / 序列并行 | 支持 |
| 专家并行 | 支持 |
| 流水线并行 | 支持 |

## 数值一致性

一致性脚本会将已发布的 Hugging Face 配置缩减，以匹配 TorchTitan 本地的 `debugmodel_mm` 配置，然后再初始化两个模型。

与 Hugging Face 实现的端到端 KL 散度（多模态输入）：**1.0212e-5**，**top-1 和 top-5 匹配率 100%**。

视觉一致性：像素预处理最大差异 **1.192e-7**；投影视觉特征的余弦相似度 **0.999987**，最大差异 **3.125e-2**。

测试脚本：

- `scripts/checkpoint_conversion/numerical_tests_kimi_k3.py` —— Hugging Face 与 TorchTitan 对比
- `tests/unit_tests/gpu/models/kimi_k3/test_kimi_k3.py` —— KDA 和 FSDP2 正确性