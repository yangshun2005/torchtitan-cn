# 模型兼容性：HF MoE 模型与 Titan MoE

本文档跟踪集成到 transformers 建模后端中的 HF MoE 模型与 Titan MoE 替换方案之间的数值等价性和架构兼容性。

## 数值等价性摘要

所有模型均使用生产级隐藏维度、缩减的专家数量（E=8）和单层配置进行测试。HF 是金标准——Titan 的 MoE 输出与未修改的 HF 前向结果进行对比。

| 模型 | model_type | 状态 | KL 散度 | max_diff | cos_sim | 往返 |
|-------|-----------|--------|--------|----------|---------|------------|
| Qwen3-30B-A3B | qwen3_moe | PASS | 5.31e-07 | 1.95e-03 | 0.999996 | PASS |
| Mixtral-8x7B | mixtral | WARN | 4.62e-05 | 3.12e-02 | 0.999996 | PASS |
| DeepSeek-V3 | deepseek_v3 | WARN | 3.66e-04 | 1.25e-01 | 0.999996 | PASS |
| OLMoE-1B-7B | olmoe | PASS | 4.04e-07 | 3.91e-03 | 0.999996 | PASS |
| DeepSeek-V2-Lite | deepseek_v2 | PASS | 2.64e-07 | 7.81e-03 | 0.999999 | PASS |
| GLM-4.7 | glm4_moe | WARN | 5.06e-05 | 6.25e-02 | 0.999996 | PASS |
| GLM-5 (DSA) | glm_moe_dsa | WARN | 2.67e-04 | 1.25e-01 | 0.999996 | PASS |
| Gemma-4-26B | gemma4_text | FAIL | 4.19e-03 | 7.46e-01 | 0.946 | PASS |

**阈值：** PASS（KL < 1e-6）、WARN（KL < 1e-3）、FAIL（KL >= 1e-3）。

## 已知的数值差异来源

### 调度器精度（影响所有 WARN 模型）

Titan 的 `LocalTokenDispatcher` 与 HF 之间存在三个独立差异：

1. **累加数据类型** —— Titan 的 `combine()` 在 `scatter_add` 之前将分数加权的专家输出转回 bf16，因此累加在 bf16 中进行；HF 保持 f32。修复：在 scatter_add 全程保持 f32。
2. **累加顺序** —— `scatter_add` 按专家排序顺序对 top-k 贡献求和，而 HF 的 `reshape(N, K, D).sum(dim=1)` 按 token 顺序求和；不同的 f32 求和顺序产生不同结果。修复：反排序回 token 顺序并使用 reshape+sum。（对 `top_k=1` 无影响。）
3. **topk 排序顺序** —— Titan 使用 `topk(sorted=False)`，HF 默认 `sorted=True`，因此路由分数归一化求和针对不同顺序的值。修复：使用 `sorted=True`。（对 `top_k=1` 无影响。）

应用全部三项修复后，所有 PASS/WARN 模型产生 max_diff=0.00。

### 激活函数（Gemma4）

Titan 默认的路由专家 `SwiGLU` 使用 SwiGLU，而 Gemma4 使用 `F.gelu(approximate="tanh")`（GeGLU）。提供模型特定的激活函数后，在相同路由下可产生精确的专家输出匹配。

### 路由器架构（Gemma4）

Gemma4 的路由器包含输入 RMSNorm、学习的 `scale` 向量以及 `per_expert_scale` 逐专家缩放——Titan 的 `TokenChoiceTopKRouter` 均不支持。这导致完全不同的路由决策，是 Gemma4 FAIL 级差异的主要来源。

## 各模型的架构差异

### Qwen3-30B-A3B (qwen3_moe)
- **路由器：** softmax，norm_topk_prob=True —— 完全支持
- **专家：** 融合 gate_up_proj (E, 2*I, H)，SwiGLU —— 完全支持
- **共享专家：** 无
- **与 Titan 的差异：** 仅调度器精度

### Mixtral-8x7B (mixtral)
- **路由器：** softmax，始终归一化（无配置标志）—— 支持
- **专家：** 融合 gate_up_proj，SwiGLU —— 完全支持
- **共享专家：** 无
- **与 Titan 的差异：** 仅调度器精度

### DeepSeek-V3 (deepseek_v3)
- **路由器：** sigmoid 配合 e_score_correction_bias，分组路由 —— 支持
- **专家：** 融合 gate_up_proj，SwiGLU —— 完全支持
- **共享专家：** 加性（DenseFeedForward）—— 支持
- **注意力：** MLA（q_lora_rank、kv_lora_rank）—— 通过 ShardingConfig TP 计划支持
- **与 Titan 的差异：** 仅调度器精度

### OLMoE-1B-7B (olmoe)
- **路由器：** softmax，norm_topk_prob=False —— 支持
- **专家：** 融合 gate_up_proj，SwiGLU —— 完全支持
- **共享专家：** 无
- **与 Titan 的差异：** 仅调度器精度

### DeepSeek-V2-Lite (deepseek_v2)
- **路由器：** softmax，无分组路由 —— 支持
- **专家：** 融合 gate_up_proj，SwiGLU —— 完全支持
- **共享专家：** 加性（2 个共享专家）—— 支持
- **注意力：** MLA —— 支持
- **与 Titan 的差异：** 仅调度器精度

### GLM-4.7 (glm4_moe)
- **路由器：** sigmoid 配合 e_score_correction_bias，分组路由 —— 支持
- **专家：** 融合 gate_up_proj，SwiGLU —— 完全支持
- **共享专家：** 加性 —— 支持
- **与 Titan 的差异：** 仅调度器精度

### GLM-5 DSA (glm_moe_dsa)
- **路由器：** sigmoid 配合 e_score_correction_bias，分组路由 —— 支持
- **专家：** 融合 gate_up_proj，SwiGLU —— 完全支持
- **共享专家：** 加性 —— 支持
- **注意力：** MLA + 动态稀疏注意力（DSA）索引器 —— MLA 通过 ShardingConfig 支持；DSA 索引器在 **TP 下不支持**（其 no_grad 前向使用 scatter_/index 操作，尚无显式本地 SPMD 覆盖）。可在无 TP 的 FSDP/EP 下运行。
- **与 Titan 的差异：** 仅调度器精度

### Gemma-4-26B (gemma4_text)
- **路由器：** softmax 配合输入 RMSNorm、学习的 scale、per_expert_scale —— **不完全支持**（Titan 路由器缺少这些组件，导致不同的路由决策）
- **专家：** 标准布局，但使用 `gelu_pytorch_tanh` 激活 —— **不支持**（Titan 硬编码 `F.silu`）
- **共享专家：** 密集 MLP 作为共享专家（层级 MoE）—— 支持（在 Titan MoE 中映射为 shared_experts）
- **MoE 架构：** 层级（路由器/专家与密集 MLP 为兄弟节点）—— 支持
- **注意力：** GQA 配合 q_norm、k_norm、v_norm —— 支持
- **与 Titan 的差异：** 激活函数不匹配 + 路由器架构不匹配（FAIL 级差异）

## 并行支持

| 模型 | FSDP | FSDP+EP |
|-------|------|---------|
| Qwen3 MoE | PASS | PASS |
| Mixtral | PASS | PASS |
| DeepSeek-V3 | PASS | PASS |
| OLMoE | PASS | PASS |
| DeepSeek-V2 | PASS | PASS |
| GLM-4.7 | PASS | PASS |
| GLM-5 DSA | PASS | PASS |
| Gemma4 | PASS | PASS |

## 完全支持所需的核心变更

### 模型特定的专家激活（针对 Gemma4）
为路由专家配置 GeGLU `ActivationFn` 实现（`gelu_pytorch_tanh`），替代默认的 SwiGLU 实现。

### 扩展路由器功能（针对 Gemma4）
`TokenChoiceTopKRouter` 需要支持输入归一化、学习的 scale 向量和逐专家缩放，以匹配 Gemma4 的路由器。