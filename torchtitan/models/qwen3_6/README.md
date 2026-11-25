# Qwen3.6

这是 Qwen3.6 检查点专用的轻量版本包。模型架构和实现仍位于 `torchtitan/models/qwen3_5/`，因为 Qwen3.6 使用与 Qwen3.5 相同的 Hugging Face 架构。

本目录仅负责：

- Qwen3.6 模型变体；
- Qwen3.6 检查点特有行为的文档。

共享的模型、Gated DeltaNet、RoPE、视觉编码器、分片、并行化及 state-dict 适配器沿用现有的 Qwen3.5 名称。

## 支持的并行方式

Qwen3.6 复用 Qwen3.5 的模型实现和并行化（见上文说明），因此支持与 [Qwen3.5 支持的并行方式](../qwen3_5/README.md#supported-parallelisms) 相同的并行方式：

| 特性 | 说明 |
|---------|-------|
| FSDP / HSDP | 解码器按层分片；视觉编码器作为单个单元分片（一次 AllGather） |
| 张量并行（TP） | 配合序列并行；GatedDeltaNet 投影上使用按头分片的 TP |
| 专家并行（EP） | 用于 `35B-A3B` MoE 变体 |
| 流水线并行（PP） | 视觉编码器分配到第一阶段；支持 1F1B 和 Interleaved1F1B 调度 |
| 上下文并行（CP） | GatedDeltaNet 使用 Attention Gym 原生 CP；全注意力支持连续、头尾或 PTRR 分区的 K/V all-gather，以及连续分区的 Ulysses |
| 样本打包 | 通过 `MMSamplePackingConfig` 选择性启用 |

## 模型变体

| 变体 | 类型 | LLM 维度 | 层数 | 头数 | KV 头数 | 专家数 | Top-k |
|---|---|---:|---:|---:|---:|---:|---:|
| `27B` | Dense | 5120 | 64 | 24 | 4 | - | - |
| `35B-A3B` | MoE | 2048 | 40 | 16 | 2 | 256 | 8 |

两个已发布变体均包含 Qwen3.5 视觉编码器。已发布的 MTP 权重不会导入或导出，因为当前模型路径仅训练主 next-token 解码器。

检查点适配器不支持预量化 FP8 仓库；请使用非 FP8 检查点作为转换输入。

## Hugging Face 资源

原训练配置分别期望资源位于 `assets/hf/Qwen3.6-27B` 和 `assets/hf/Qwen3.6-35B-A3B`。这些路径仍可作为用户自有配方的起点，但 TorchTitan 不为任一变体提供未经验证的训练配方。