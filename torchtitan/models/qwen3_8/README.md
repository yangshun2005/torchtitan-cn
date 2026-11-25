# Qwen3.8

这是面向 Qwen3.8 检查点的轻量级版本专用包。模型架构和实现仍位于 `torchtitan/models/qwen3_5/`，因为 Qwen3.5 和 Qwen3.8 使用相同的 Qwen3.5 Hugging Face 架构。

本目录仅负责：

- Qwen3.8 模型变体；
- Qwen3.8 检查点特定行为的文档。

共享的模型、Gated DeltaNet、RoPE、视觉编码器、分片、并行化以及 state-dict 适配器沿用现有的 Qwen3.5 命名。

## 支持的并行方式

Qwen3.8 复用 Qwen3.5 的模型实现和并行化（见上方说明），因此支持与 [Qwen3.5 支持的并行方式](../qwen3_5/README.md#supported-parallelisms) 中记录的相同并行方式：

| 特性 | 说明 |
|---------|-------|
| FSDP / HSDP | 解码器按层分片；视觉编码器作为单个单元分片（一次 AllGather） |
| 张量并行（TP） | 配合序列并行；GatedDeltaNet 投影上使用按头分片的 TP |
| 专家并行（EP） | 用于 `2.4T-A95B` MoE 变体 |
| 流水线并行（PP） | 视觉编码器分配到第一阶段；1F1B 和 Interleaved1F1B 调度 |
| 上下文并行（CP） | GatedDeltaNet 使用 Attention Gym 原生 CP；全注意力支持连续、头尾或 PTRR 分区的 K/V all-gather，以及连续分区的 Ulysses |
| 样本打包 | 通过 `MMSamplePackingConfig` 选择性启用 |

## 模型变体

| 变体 | 类型 | LLM 维度 | 层数 | 头数 | KV 头数 | 专家数 | Top-k | 视觉 |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `27B` | 稠密 | 5120 | 64 | 24 | 4 | - | - | 是 |
| `2.4T-A95B` | MoE | 8192 | 92 | 64 | 4 | 512 | 10 | 否 |

Qwen3.8-27B 在架构上与 Qwen3.5-27B 完全相同。2.4T-A95B 检查点扩展了相同的 MoE 解码器，并将其纯文本语言模型权重存储在 `model.*` 下，而非多模态的 `model.language_model.*` 前缀。

Hugging Face 配置新增了 `output_gate_type: "swish"`；这与 TorchTitan 的 Gated DeltaNet 中现有的 SiLU/Swish 门控一致。全注意力输出门控仍为 sigmoid。已发布的 MTP 权重不会被导入或导出，因为此模型路径当前仅训练主 next-token 解码器。

检查点适配器不支持预量化的 FP8 仓库；请使用非 FP8 检查点作为转换输入。

## Hugging Face 资产

此前的训练配置分别期望资产位于 `assets/hf/Qwen3.8-27B` 和 `assets/hf/Qwen3.8-2.4T-A95B`。这些路径仍可作为用户自有配方的有用起点，但 TorchTitan 不为这两种变体提供未经验证的训练配方。