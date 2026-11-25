# Qwen3.5：混合注意力多模态模型

## 概述

Qwen3.5 组合了：
- **混合解码器**：75% GatedDeltaNet（线性注意力）+ 25% 全注意力，带输出门控和部分 RoPE。
- **视觉编码器**：Vision Transformer（ViT），使用 2D RoPE 和双线性插值的学习位置嵌入。
- **Patch Merger**：合并空间 patch 以缩短视觉序列长度（如 2x2 patch -> 1 个 token）。
- **MRoPE**：在解码器层中交错时间、高度、宽度位置 ID 的 RoPE。
- **MoE 变体**：路由专家 + 共享专家，带 sigmoid 门控。

## 视觉散射

- `tok_embeddings` 生成形状为 `B×S` 的文本 token 嵌入。
- `vision_encoder` 生成形状为 `N×L` 的视觉 token 嵌入。
- 有效视觉 token（不含 padding）被散射到文本序列的占位位置，如下图所示（来源：[@lkhphuc](https://github.com/lkhphuc)）。

<img width="1398" height="840" alt="VLM 架构" src="https://github.com/user-attachments/assets/63fcbbc1-c587-4a63-8246-411cb72f5789" />

注意：图中每个 patch 映射到一个视觉 token。实际中 Patch Merger 将 `merge_size²` 个 patch 合并为一个 token（如 `merge_size=2` → 每 token 4 个 patch），视觉序列长度缩短 `merge_size²` 倍。

## 前置条件

安装额外依赖：

```bash
pip install -r .ci/docker/requirements-vlm.txt
```

## 模型变体

以下已发布规格在完成端到端验证前，有意不提供训练配方。
用户自有的配方可使用对应的 `assets/hf/Qwen3.5-<variant>` 目录；例如规格
`35B-A3B` 按惯例使用 `assets/hf/Qwen3.5-35B-A3B`。

上述每个规格名均为纯文本模式：构建 `Qwen35Model`，即不含视觉编码器的解码器，
并从多模态检查点读取语言模型。追加 `_mm`（如 `9B_mm`、`35B-A3B_mm`）则构建
`Qwen35MultimodalModel`，额外包含视觉编码器。

### 稠密

| 变体 | LLM 维度 | 层数 | 头数 | KV 头数 | 头维度 | ViT 维度 | ViT 层数 |
|---------|---------|--------|-------|----------|----------|---------|------------|
| debugmodel | 256 | 8 | 4 | 2 | 64 | 256 | 4 |
| 0.8B | 1024 | 24 | 8 | 2 | 256 | 768 | 12 |
| 2B | 2048 | 24 | 8 | 2 | 256 | 1024 | 24 |
| 4B | 2560 | 32 | 16 | 4 | 256 | 1024 | 24 |
| 9B | 4096 | 32 | 16 | 4 | 256 | 1152 | 27 |
| 27B | 5120 | 64 | 24 | 4 | 256 | 1152 | 27 |

### MoE

| 变体 | LLM 维度 | 层数 | 专家数 | Top-k | 共享专家 |
|---------|---------|--------|---------|-------|---------------|
| debugmodel_moe | 256 | 4 | 8 | 2 | 是 |
| 35B-A3B | 2048 | 40 | 256 | 8 | 是 |
| 122B-A10B | 3072 | 48 | 256 | 8 | 是 |
| 397B-A17B | 4096 | 60 | 512 | 10 | 是 |

## 数据集

| 数据集 | 类型 | 格式 |
|---------|------|--------|
| `cc12m` | 图文对 | WebDataset（流式） |
| `cc12m-test` | 图文对 | 本地 WebDataset（用于测试） |

## 支持的并行方式

| 特性 | 说明 |
|---------|-------|
| FSDP / HSDP | 解码器按层分片；视觉编码器作为整体分片（一次 AllGather） |
| 张量并行（TP） | 配合序列并行；GatedDeltaNet 投影上按头分片 TP |
| 专家并行（EP） | 用于 MoE 变体 |
| 流水线并行（PP） | 视觉编码器分配到第一阶段；支持 1F1B 和 Interleaved1F1B 调度 |
| 上下文并行（CP） | GatedDeltaNet 使用 Attention Gym 原生 CP；全注意力支持 K/V all-gather，支持连续、头尾或 PTRR 分区，以及连续分区的 Ulysses。Ulysses 头数规则见 `UlyssesInnerAttention`。 |
| 样本打包 | 通过 `MMSamplePackingConfig` 选择性启用 |

## 数值一致性

与 HuggingFace Transformers 的端到端 KL 散度（4B，多模态输入）：平均 **~3e-7**，**top-1 和 top-5 匹配率 100%**。

并行正确性：无并行、FSDP、FSDP+EP、FSDP+EP+TP 配置下 logits 逐位一致（最大差异 `0.0`）。

测试脚本：
- `scripts/checkpoint_conversion/numerical_tests_qwen3_5.py` — HF 与 TT 对比
- `scripts/checkpoint_conversion/numerical_tests_qwen3_5_shard.py` — 并行正确性

## TODO

- 添加视频数据集训练配置