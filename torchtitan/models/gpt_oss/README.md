# torchtitan 中的 gpt-oss 模型

## 快速开始
```bash
MODULE=torchtitan_recipes.tests.models.gpt_oss CONFIG=gpt_oss_debugmodel ./run_train.sh
```

已发布的 20B 和 120B 版本按惯例使用
`assets/hf/gpt-oss-20b` 和 `assets/hf/gpt-oss-120b`。TorchTitan 保留这些
资源路径提示，但不提供未经核实的完整训练配方。

## 支持的特性
- FSDP/HSDP、TP、EP、CP、PP
- 分组矩阵乘法，实现高效计算

使用 DeepEP 且不做专家输出后处理时，路由分数会缩放 W2 的输入
及其每专家偏置。下式中，`b2` 沿路由 token 维度展开：每行获得其
所选专家的偏置，`s` 在该行的特征维度上广播。

```text
s * (W2(h) + b2) = W2(s * h) + s * b2
```

输入缩放和偏置缩放以 FP32 相乘，再转回投影的数据类型。
W2 及其偏置加法共享 `grouped_mm` 重算策略。若在保存 W2 后重放哪怕
廉价的偏置加法，也需要保留 W2 的完整路由输出；同时保存两者可避免
该依赖。偏置缩放增加了逐点运算和归约，GEMM 次数与无偏置吸收相同。
若启用专家输出后处理，则改为在后处理之后应用分数。其他 dispatcher
仍保留专家后打分。该等式在精确算术下成立；移动分数位置会改变
BF16 舍入，可能改变输出和梯度。

CI 已在 debug 模型上运行 CP 和 PP：
- `gpt_oss_pp+fsdp+cp+ep+sacop`（`gpt_oss_debugmodel_flex_fsdp2_cp2_pp2_ep4_sac`）
- `gpt_oss_pp+fsdp+ep+sacop`

这些任务使用 Interleaved1F1B。FlexInnerAttention 零气泡 / 拆分反向 PP
测试在 `tests/integration_tests/features.py` 中被禁用，因为
FlexInnerAttention 的 `BlockMask` 不是 Tensor（`stage_backward_input` 会对
每个 stage 输入调用 `requires_grad`）。全反向调度
（1F1B / GPipe / Interleaved1F1B）不受影响。