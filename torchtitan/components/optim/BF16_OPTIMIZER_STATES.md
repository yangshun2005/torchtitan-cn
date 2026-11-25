## BF16 优化器状态

在默认的 fp32 训练配置（`precision.dtype="float32"`）下，Adam/AdamW 以 float32 保存动量（`exp_avg`）和方差（`exp_avg_sq`），相比以 bfloat16 存储这些缓冲区，优化器状态内存大约翻倍。

将 Adam/AdamW 优化器配置的 `moment_dtype` 设为 **`"bfloat16"`**，即可使用融合 CUDA 内核，实现 **bf16 优化器动量** 与 **fp32 参数**。这会降低优化器状态内存，同时保持参数更新为 fp32。FSDP 仍可能以 bf16 规约梯度，它会在优化器步进前将规约后的梯度分片转换为 fp32。

如果使用 **`precision.dtype="bfloat16"`**（参数和梯度均为 bf16），请保持 `moment_dtype="parameter"`（默认值）。此时 PyTorch 会让优化器状态 dtype 与训练对齐。

这适用于内存受限的训练场景，且动量估计精度略有降低可以接受。

### 背景

该技术曾被 [DeepSeek-V3](https://arxiv.org/abs/2412.19437) 用于在 14.8 万亿 token 上训练其 671B 参数的 MoE 模型，并降低了内存开销。他们的方法证明，动量和方差缓冲区均可存为 bfloat16 而不引发收敛问题，尤其适用于专家梯度幅值较小的 MoE 架构。为 PyTorch 添加原生 bf16 AdamW 支持的工作见 [pytorch/pytorch#146542](https://github.com/pytorch/pytorch/issues/146542)。

### 用法

要使用 BF16 优化器状态与 FP32 持久参数，请使用：

```python
config.precision.dtype = "float32"
config.precision.param_dtype = "bfloat16"
config.precision.reduce_dtype = "float32"
# config.precision.reduce_dtype = "bfloat16"  # 可选 BF16 规约。
config.optim.optimizer.optimizers[0].moment_dtype = "bfloat16"
```

规约 dtype 与优化器状态 dtype 相互独立。FP32 梯度规约请保持 `reduce_dtype="float32"`。设为 `"bfloat16"` 可降低 FSDP 通信带宽。使用 BF16 规约时，FSDP 会在优化器步进前将每个规约后的梯度分片转换回 FP32 参数 dtype。

### 要求

- **优化器**：必须为 `Adam` 或 `AdamW`。
- **实现**：优化器配置必须设置 `fused=True`。融合 CUDA 内核（`FusedAdamMathFunctorMP`）处理混合精度更新（fp32 参数 + bf16 动量）。配置阶段会拒绝 for-loop 或 foreach 实现搭配 bf16 动量。

这些约束在配置时校验。

### 工作原理

每个优化器实例上注册了一个步进前置 Hook。在第一步 Adam 惰性状态初始化运行之前，该 Hook 预先将 `exp_avg` 和 `exp_avg_sq` 填充为 bfloat16 张量。当 `_init_group` 发现状态非空时，会跳过自身的 fp32 分配。融合内核检测到 fp32 参数与 bf16 状态之间的 dtype 不匹配，并分派到混合精度代码路径。

### 与其他特性的交互

- **`precision.dtype`**：主要用例是 `float32` 训练搭配 `moment_dtype="bfloat16"` 以节省优化器状态内存。使用 `bfloat16` 训练时，默认的 `moment_dtype="parameter"` 通常已足够；参见上文介绍。
- **检查点**：启用此选项时，优化器动量以 bfloat16 保存。恢复时请使用相同的 `moment_dtype`。检查点加载会先创建 bf16 动量，再由 DCP 加载到其中。`load_flat_optim_state_dict` 在 `torch.optim.preserve_state_dtype_on_load()` 下运行 `Optimizer.load_state_dict`，因此动量保持 bf16，不会产生 fp32 副本。`step` 计数器保留 PyTorch 的 dtype 和设备策略。跨保存/加载混用动量 dtype 仍不受支持。
- **FSDP**：兼容 FSDP2。优化器看到的是 DTensor 参数；bf16 状态 Hook 作用于本地分片。

### 限制

- 仅 TorchTitan 的 `Adam` 和 `AdamW` 封装支持 `moment_dtype="bfloat16"`。
- 优化器必须使用 `fused=True`；for-loop 和 foreach Adam 内核不支持 FP32 参数搭配 BF16 动量。
- 动量估计精度降低可能影响某些模型或超参数设置的收敛性。用户应针对具体用例验证损失收敛。