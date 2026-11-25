# 位级数值与批次不变性

本文说明 RL 训练循环中数值非确定性的来源、消除这些非确定性的机制，以及每个机制在代码中的位置。面向用户的配置与运行说明，见 `torchtitan/rl/README.md` 的“Batch-invariant mode”和“Verifying generator/trainer logprob parity”两节。

## 问题

在 LLM 的 RL 中，生成器在一种批次组成下（例如 8 条 completion）计算 completion 的 log-prob，而训练器在另一种批次组成下（例如 DP 分片后的 2 条 completion）重新计算。若不特殊处理，同一输入在不同批次上下文中会产生不同的 log-prob，原因是 GEMM、softmax、归约和集合通信操作中的浮点累加顺序不同。即使极小的漂移也可能翻转 MoE 模型的 top-k 专家路由，因此两条路径必须逐算子一致。

## 接入位置

批次不变模式由训练器和生成器配置上的 `DebugConfig(batch_invariant=True, deterministic=True)` 控制。共享开关是 `torchtitan/distributed/batch_invariant.py` 中的 `set_batch_invariance()`，由以下位置调用：

- `torchtitan/rl/trainer/trainer.py`（训练器）
- `torchtitan/rl/generator/generator.py`（生成器）
- `torchtitan/rl/generate.py`（独立生成）

## 批次不变性的破坏来源

对同一输入，训练器和生成器必须以相同累加顺序运行相同算子。三组修复共同保证这一点。

**共享算子覆盖** —— 由两个角色都调用的 `set_batch_invariance()`（`torchtitan/distributed/batch_invariant.py`）应用：

- **`mm` / `addmm` / `_log_softmax` / `mean.dim`：** cuBLAS 和默认归约内核根据输入形状选择 tile/block 调度，因此累加顺序（及舍入）随批次大小变化。改用上游 [`batch_invariant_ops`](https://github.com/thinking-machines-lab/batch_invariant_ops) 的 Triton 内核覆盖，这些内核使用固定的 tile 迭代顺序。
- **Flash attention split-k：** FA3 根据序列长度选择 `num_splits`，改变部分和归约树。当 `is_in_batch_invariant_mode()` 为 True 时强制 `num_splits=1`（`torchtitan/models/common/attention/attention.py`，`VarlenInnerAttention.forward`）；FA2 本身已是批次不变的。
- **NCCL 集合通信：** all-reduce 可能使用 ring/tree 算法且通道数可变，导致跨 rank 归约顺序变化。强制为单通道 tree（`NCCL_ALGO=allreduce:tree`，`NCCL_MAX_NCHANNELS=1`）；必须在 `dist.init_process_group` 之前设置。
- **降低精度的归约和 TF32：** 禁用（`allow_bf16/fp16_reduced_precision_reduction = False`，`torch.backends.cuda.matmul.allow_tf32 = False`，`torch.backends.cudnn.allow_tf32 = False`）。

**仅生成器的补丁**（`torchtitan/rl/generator/vllm/batch_invariance.py`）—— 生成器在 vLLM 内运行同一模型，vLLM 的融合内核必须路由回训练器的算子：

- **注意力（Varlen / Flex / GPT-OSS）：** 生成器必须运行与训练器相同的注意力内核，具体取决于配置使用的后端。
  - *Varlen*（Qwen3，默认）：生成器通过 vLLM 的 `CUSTOM` 后端运行 torchtitan 自己的 `VarlenInnerAttention` —— 与训练器相同的 FA3 varlen 内核。上述共享的 `num_splits=1` 修复消除了唯一依赖批次的 split-k，因此两者位级一致。
  - *Flex*（Qwen3 flex）：生成器运行 vLLM 的 `FLEX_ATTENTION` 后端。`BatchInvariantFlexTransform`（recipe 同时应用于训练器和生成器模型配置的 `ModelConfigTransform`）在每个 `FlexInnerAttention` 层上固定 `BLOCK_M = BLOCK_N = 16` 和 `BACKEND = "TRITON"`，使两侧使用相同的 Triton tile 大小并避开 `flex_decode` 内核。
  - *GPT-OSS*（varlen + 逐层滑动窗口 + attention sinks）：与 Varlen 相同的 `CUSTOM` varlen 路径和 `num_splits=1`，滑动窗口大小写入每层的 `VarlenInnerAttention` 配置（两侧相同）。attention sinks 是 softmax 后的重缩放（`apply_attention_sink_rescale`，`sigmoid(lse - sinks)`）；vLLM 包装器的 `_inject_attention_sinks` 将同一重缩放安装为生成器注意力的 `out_transform`，使 sink 计算与训练器一致。
- **`force_trainer_logprobs_for_batch_invariance`：** vLLM 的 v2 GPU sampler 用融合 Triton 内核计算逐 token logprob，该内核内联 `log(softmax(logits))` 且从不调用 PyTorch 算子。改为训练器的 `compute_logprobs`，使两者共享同一 logprob 代码路径。

**RoPE 缓存 dtype（`freqs_cis`）：** `freqs_cis` 以 fp32 存储，并作为前向输入传给每个 `TransformerBlock`。FSDP2 的 `MixedPrecisionPolicy(cast_forward_inputs=True)` 会在每个 FSDP 边界将其转为 bf16，导致 `apply_rotary_emb_cos_sin` 中的 bf16 `cos`/`sin` 丢失精度（乘法会提升回 fp32，但 bf16 舍入已经发生）。torchtitan 在标准 `apply_fsdp`（`torchtitan/distributed/fsdp.py`）中设置 `cast_forward_inputs=False`，使 `freqs_cis` 在所有路径中保持 fp32 —— 这是跨并行配置（例如仅 TP vs FSDP+TP）位级一致数值的必要条件。对浮点输入已是 `param_dtype` 的 LLM 安全；带外部 fp32 输入（例如像素值）的图像模型（Flux、VLM）可能仍需 `cast_forward_inputs=True`。

## 精度要求

批次不变模式要求训练器和生成器都使用 bfloat16 计算，以保持相同精度（`precision.param_dtype="bfloat16"`，默认值）。生成器以 `precision.param_dtype` 计算。训练器通过 FSDP 混合精度达到相同精度：保留 fp32 主权重，FSDP 的 `MixedPrecisionPolicy` 在前向前将其转为 bf16，因此前向与生成器的 bf16 路径数值一致。`preserve_dtype_params` 中列出的参数在两侧保持主 dtype。训练器始终用 FSDP 包装模型（`Decoder._apply_fsdp` 为训练器应用 `apply_fsdp_to_decoder`），因此即使在 `data_parallel_shard_degree=1` 时也会发生此转换，此时 FSDP 仅作为混合精度边界（degree-1 的 all-gather 是空操作但仍转为 bf16）。相对于仅 TP 不需要额外 GPU。当 `data_parallel_shard_degree > 1` 时，同样的转换在分片 all-gather 期间发生。

批次不变配置保留 fp32 主权重（`PrecisionConfig()`，默认 `dtype="float32"`），并依赖 FSDP 混合精度进行 bf16 前向，因此优化器更新 fp32 权重，而前向与生成器位级一致。

## 当前限制

- 训练器和生成器必须使用相同的张量并行度（logprob 位级一致性仅在并行度匹配时支持）。
- 批次不变模式不支持序列并行，因为 reduce-scatter 缺少基于 tree 的实现，而基于 ring 的实现是非确定性的。

## 用法

通过 RL 入口运行批次不变配置，例如：

```bash
python -m torchtitan.rl.train \
    --module torchtitan_recipes.rl.alphabet_sort \
    --config rl_grpo_qwen3_0_6b_varlen_batch_invariant
```

`torchtitan_recipes/rl/alphabet_sort.py` 及其调试模型对应文件 `torchtitan_recipes/tests/rl/alphabet_sort.py` 中的批次不变配置，在训练器上都使用 FSDP 混合精度（fp32 主权重，bf16 转换前向）：

## 性能：批次不变模式的成本

批次不变模式换入确定性内核（`batch_invariant_ops` 的 Triton mm/addmm/log_softmax，以及强制 `num_splits=1` 的 flash attention），并禁用 TF32 + 降低精度归约，因此原始计算更慢。这种减速是否体现在端到端取决于工作负载的计算密集程度。

**实验设置。** Search-R1 recipe 上的 Dense Qwen3-8B（`rl_grpo_qwen3_8b_search_r1`，带与不带 `_batch_invariant` 变体），8xH100，匹配 TP2/TP2（训练器 `tensor_parallel_degree=2` + 生成器 `tensor_parallel_degree=2`），on-policy（`target_offpolicy_steps=0`），两次运行均开启生成器 CUDA graph（`FULL_DECODE_ONLY`），30 个训练步，4 组 x group_size 8 = 每步 32 条 rollout。两次运行仅在 `DebugConfig(batch_invariant=...)` 上不同。所有值均为每步中位数。

| 指标（中位数）                 | BI 关闭             | BI 开启     | BI 的影响       |
|-------------------------------|--------------------|-----------|-----------------|
| 每步墙钟时间                   | 136.4 s            | 133.6 s   | ~1.0x（无成本） |
| 生成器 ITL（每 token）         | 9.6 ms             | 22.8 ms   | 慢 2.4x         |
| 生成器解码时间                 | 177 ms             | 460 ms    | 慢 2.6x         |
| 训练器前向/反向吞吐量          | 370 tok/s          | 127 tok/s | 慢 2.9x         |
| 训练器全步吞吐量               | 11.1 tok/s         | 9.2 tok/s | 慢 1.2x         |
| kl_div 最大值 / logprob_diff 最大值 | 0.0084 / 1.56 (>0) | 0 / 0     | 位级一致        |

**要点。**

- 批次不变模式使原始*计算*慢约 2.4-2.9x（生成器每 token 延迟和训练器前向/反向），这是确定性内核的成本。
- 对于此工作负载，每步端到端墙钟时间基本不变（~1.0x）：Search-R1 的一步主要由非计算工作主导 —— 多轮 rollout 编排、CPU 检索往返，以及等待最慢落后者 rollout 的 on-policy 同步 —— 这掩盖了计算减速。在计算密集型工作负载上（长生成、少量检索/编排），~2.5x 会体现在墙钟时间上。
- 这一成本换来训练器/生成器的精确一致：BI 开启时每步 `kl_div` 和 `logprob_diff` 均为 0，而 BI 关闭时 > 0（最高 0.008 / 1.56）。