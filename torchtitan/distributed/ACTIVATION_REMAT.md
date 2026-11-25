# `torch_remat` 激活重计算

激活重计算（remat），也称激活检查点，用计算换内存：Transformer 块在前向中只保留部分激活，反向时重算其余部分。TorchTitan 的重计算策略使用 `torch_remat` 对每个 Transformer 块做检查点。保存区域之外的操作在反向时重算。

## 动机

模型代码标识语义上的计算与通信区域，重计算策略则选择保留哪些区域的输出。

这需要在模型代码中写少量显式标注。作为回报，策略面就紧挨着它所控制的操作，配置也引用稳定的模型概念（如注意力投影），而非单个 ATen 算子。

策略是配置对象，设为训练器的 `activation_remat` 字段，并通过 `config.apply(model)` 应用到每个 Transformer 块：

- `ActivationRematConfig` 保存匹配 `save_regions` 的区域（匹配 `recompute_regions` 的除外），重算其余全部。
- `FullActivationRematConfig` 是固定策略，`save_regions=[]`。它重算块内每个操作，但模型代码以 `recompute=False` 始终保留的区域除外，例如路由决策和尾部加法。
- `DefaultActivationRematConfig` 是训练器的默认策略：`save_regions=["*"]`、`recompute_regions=["*routed_experts.w13.*"]`。其区域可能随测量改进而变化；需要固定策略时请用带显式区域的 `ActivationRematConfig`。它保留模型声明的每个区域，但路由专家的 `w13` 分组投影除外，并重算这些区域之外的操作。路由专家分组 matmul 的激活随 top-k 增长，主导 MoE 激活内存。`w2` 保持保存：其保存的输入就是激活输出，重放本就会重建它，重算它只会耗时而不省内存。EP token 分发器通信区域保持保留，因为旧策略保存了全部 all-to-all 输出。未声明任何区域的模型在此策略下会全量重算。

原算子级 SelectiveAC 策略及其 `force_recompute_mm_shapes_by_fqns` 选项已移除。策略需要比固定策略更细的控制时，请用 `ActivationRematConfig`。

## 配置保存区域

`ActivationRematConfig.save_regions` 包含相对于 Transformer 块的 shell 风格模式。例如：

```python
ActivationRematConfig(
    save_regions=[
        "attention.qkv_linear.wqkv.linear",
        "attention.wo.linear",
    ]
)
```

`ActivationRematConfig.recompute_regions` 使用相同的模式语法，且优先于 `save_regions`。从无 AC 开始，先保存全部，再列出要重算的少数区域，直到模型适配内存预算：

```python
ActivationRematConfig(
    save_regions=["*"],
    recompute_regions=["moe.routed_experts.w13.grouped_mm"],
)
```

支持 `attention.*` 这类通配符。未匹配的模式目前会被忽略；校验最终必须覆盖所有流水线阶段的区域。

### 按块策略

策略默认应用到每个 Transformer 块，除非 `block_overrides` 对选定块做了修改。统一策略在每个块付出相同代价：若在两个块中重算一个昂贵区域就足以适配，统一选择就是在所有块中重算它。覆盖项指定块，并替换这些块的 `save_regions`、`recompute_regions`、`offload_block_input` 和 `offload_regions` 中的任意项；留为 `None` 的字段沿用基础值：

```python
ActivationRematConfig(
    save_regions=["*"],
    block_overrides=[
        # Recompute the attention core only in the first two blocks.
        BlockRematOverride(
            blocks=["layers.0", "layers.1"],
            recompute_regions=["attention.inner_attention"],
        ),
    ],
)
```

块名是原始模型中的精确 FQN，格式为 `<container fqn>.<id>`：解码器块为 `layers.<id>`，多 token 预测块为 `mtp_layers.<id>`，编码器块为 `<encoder>.layers.<id>`（如 `vision_encoder.layers.3`）。每个模型在 `BaseModel._remat_block_container_fqns` 中列出其块容器，`parallelize` 将它们传给 `apply(model, block_container_fqns=...)`。流水线切分保留原始层 id，因此覆盖项会跟随其块到所在阶段。空覆盖项和同一块被两个覆盖项指定会在配置时报错；模型缺少的名字不会被检测到，因为每个流水线阶段只持有部分块。在块的结果内，`recompute_regions` 仍优先于 `save_regions`，因此只设置 `save_regions` 的覆盖项无法保留基础策略要重算的区域；需同时覆盖 `recompute_regions`。固定预设接受只设置 offload 字段的覆盖项；要改变它们保留的内容，请用 `ActivationRematConfig`。

## 保存张量 Hook

被检查点的块在前向到反向之间保持少量张量存活：其输入（重算重放所依据）以及保存区域保留的张量。torch_remat 的保存张量 Hook 正好看到这些：`pack` 将每个张量替换为载荷，`unpack` 在重算或反向需要时重建它。在 `pack` 内，`torch_remat.current_saved_tensor_info().kind` 说明张量被保留的原因：块输入为 `CHECKPOINT_INPUT`，保存区域保留的张量为 `SAVE_OUTPUT` 和 `BACKWARD`。`pack` 在块入口运行，早于前向读取其输入，因此不得释放或覆盖其存储；unpack 不必按 pack 的逆序运行；检查点在 `torch.compile` 下运行时，torch_remat 会拒绝这些 Hook。参见 torch_remat 的[卸载指南](https://github.com/meta-pytorch/remat/blob/main/docs/offloading.md)。

TorchTitan 用它们做 CPU 卸载，由重计算策略驱动：`offload_block_input` 和 `offload_regions` 选择哪些保留张量送往 CPU，随后 `apply` 安装卸载器，由它与计算重叠执行拷贝。参见 [ACTIVATION_OFFLOAD.md](ACTIVATION_OFFLOAD.md)。

## 诊断生效策略

应用重计算策略后，在要检查的前向周围使用 `torch_remat` 的跟踪收集器：

```python
import torch_remat as remat

with remat.collect_trace() as trace:
    output = model(inputs, **model_kwargs)

print(trace.format())
```

例如，跟踪可能如下：

```text
torch_remat trace
attention.qkv_linear.wqkv.tp_gather: recompute
attention.qkv_linear.wqkv.linear: save
attention.inner_attention: recompute
attention.wo.linear: save
attention.wo.tp_reduce: save
feed_forward.w13.tp_gather: recompute
feed_forward.w13.linear: recompute
feed_forward.w2.linear: save
feed_forward.w2.tp_reduce: save
```

跟踪按执行顺序列出实际执行的区域，以及每个区域是保存还是重算。全模型前向可能包含来自不同 Transformer 块的重复区域名。此诊断由调用方显式控制，因此可限定到所调查的模型输入、批次或块，而无需改动训练配置。

## 在模型代码中添加区域

模型代码在被控制的操作处定义区域：

```python
out = remat.region(
    self.inner_attention,
    self.remat_region_name("inner_attention"),
    recompute=self.remat_should_recompute("inner_attention"),
)(q, k, v)
```

重计算策略用模块相对于 Transformer 块的名称及其保存模式来配置每个模块。因此上述辅助函数将 `inner_attention` 解析为 `attention.inner_attention` 这样的限定名，并选择保存还是重算。没有外层 `remat.checkpoint` 时，`remat.region` 不改变执行。

每个 `Linear` 声明自己的区域，因此模型代码直接调用它：

- `<fqn>.linear` 是本地投影。它覆盖所有 `Linear` 子类，包括量化和 LoRA linear，后者只覆盖本地计算。LoRA 适配器在基础投影的区域内运行，而不声明自己的区域，因为嵌套在保存区域内的区域无法重算。
- `GroupedLinear` 以同样方式在其分组 matmul 周围声明 `<fqn>.grouped_mm`，例如 `moe.routed_experts.w13.grouped_mm`。
- `ColumnParallelLinear` 在投影前添加 `<fqn>.tp_gather`：序列并行下是输入 all-gather，否则是前向空操作、反向做 all-reduce。保存投影而重算 gather 只保留序列分片：反向为权重梯度重放 all-gather，而不保留已 gather 的输入。
- `RowParallelLinear` 在投影后添加 `<fqn>.tp_reduce`，单独控制。保存 `linear` 而重算 `tp_reduce` 会为重放的归约保留大 TP 倍的部分输出；两者都保存可避免。

例如，融合注意力投影是 `attention.qkv_linear.wqkv.linear`。不要将 `Linear` 调用包在另一个区域中：保存的外层区域不能包含被重算的内层区域。

当多个普通 `Linear` 投影共享一个 TP 输入时，模块在它们的公共边界用 `maybe_gather_tp_input(self, x)` 只 gather 一次，它声明 `<module fqn>.tp_gather`，语义与 `ColumnParallelLinear` 的 gather 相同。每个注意力模块将其内核包在 `<module fqn>.inner_attention` 中。

## 声明重算依赖

在原始前向中，`torch_remat` 判断保存区域的输出在重算期间是否需要。当输出被显式的 `remat.region(..., recompute=True)` 消费时，它可以推断此依赖。

若消费者不在这样的区域内，请在输出被消费前立即调用 `remat.recompute_needs_tensor(...)`：

```python
gate_up = self.w13(x)  # declares feed_forward.w13.linear
gate, up = gate_up.unbind(-2)
remat.recompute_needs_tensor(gate, up)
hidden = F.silu(gate) * up
```

没有此标记，普通重算操作所需的张量可能不会被保留。请将标记放在消费者一侧，紧挨读取该张量的裸操作之前，而不是紧跟在产生它的区域之后。这样可确保仅当该消费者实际运行时才保留输出。可以传入视图，因为 `torch_remat` 会按存储将其解析到产生它的区域。视图操作本身（`view`、`unbind`、`split`、`transpose`、不拷贝的 `contiguous` 或 `reshape`）无需标记，即使在保存区域的输出上：重放视图只涉及元数据。

当一个裸操作消费多个区域输出时，将它们全部传给一次调用，如上例。不同消费者请分开调用。若输出只被另一个 `remat.region` 消费，不要添加标记；该依赖会自动推断。

当区域输出直接从被检查点的 Transformer 块返回，且块内没有操作读取其数据时，可省略标记。仅作为子模块中的最后一个区域还不够：例如，注意力输出仍可能被外层 Transformer 块的残差加法消费。若消费者位于拥有该区域的辅助函数之外，请在模块边界允许的范围内将标记尽量靠近该调用点消费者。

### 尾部加法

`torch.utils.checkpoint` 在检查点下保存的每个张量都重算完毕后停止重放，因此它从不重跑不保存任何内容的尾部操作，例如块末尾的残差加法。`remat.checkpoint` 会重放整个块，因此裸尾部加法需要一个标记，使其输入（如 MoE combine 输出）从前向一直存活到重放。为此，块末残差加法（`ffn_residual`）和共享专家加法（`moe.shared_add`）是固定的 `recompute=False` 区域：它们不为反向保存任何内容，因此保存的加法不驻留任何内容，其输入无需标记，重放会跳过它。只包装加法；尾部 norm（如 Muse Glimmer 的 post-norm）仍是裸操作，并在其输入上带标记。

## 随机状态

所有策略都要求 `preserve_rng_state=False`。可能在保存区域内推进的随机状态必须改用显式的 `torch_remat.RecomputeStateHook` 管理。

## 确定性检查

`determinism_check` 选择如何将重算张量与原始前向比对；参见 [PyTorch 检查点文档](https://docs.pytorch.org/docs/stable/checkpoint.html)。策略不支持 `debug=True`；请用[诊断生效策略](#diagnosing-the-effective-policy)中描述的跟踪收集器检查哪些区域被保存或重算。

## 前向副作用

为日志或优化器步更新累积的状态只能在原始前向上推进。始终保留的 MoE `routing_decision` 区域负责专家选择和分位直方图观测，而 token 计数累积显式忽略检查点重放。两者复用为分发和辅助损失构建的路由映射。Kimi K2.7 QK-clipping 统计也忽略重放。辅助损失累积使用始终保留的区域。

当前支持的 Transformer 块在其前向内不推进 RNG 状态，因此不需要 `RecomputeStateHook`。未来任何 dropout、随机舍入计数器或其他外部 RNG 状态都必须先添加 Hook，才能在这些策略下安全使用。

## 保存昂贵的 MoE 工作

避免重放昂贵的 MoE 工作需要同时保留其计算和通信区域：

- 路由专家的 `w13` 和 `w2` 分组投影（`w13.grouped_mm`、`w2.grouped_mm`）。
- Token 分发器的 `dispatch` 和 `combine`。使用 all-to-all 分发器（EP > 1）时，各为一个区域：`dispatch` 覆盖专家排序、token 计数交换及其设备到主机同步、分发 all-to-all 和专家主序置换；`combine` 覆盖反置换、combine all-to-all 和分数加权 scatter-add。DeepEP 和 HybridEP 分发器则在其内核周围声明 `ep_communication.dispatch` 和 `ep_communication.combine`。两者都随 `ep_communication` 一起保存或重放 dispatch 和 combine。重放必须按前向顺序接收 token，因此 DeepEP 使用确定性缓冲区，这会给每次分发增加一次排序。HybridEP 通过前缀扫描推导 token 位置。`FullActivationRematConfig` 会重放两个后端的通信。
- 共享专家 linear 区域。启用序列并行时，共享的 `w2.tp_reduce` 区域是 `Partial -> Shard(0)` reduce-scatter；请与 `w2.linear` 一起保存。
- `tp_output_reduction`，在禁用序列并行时控制最终 TP all-reduce。在带共享专家的批次不变模式下，路由输出有自己的 TP all-reduce，即 `tp_routed_reduction`。

对于名为 `moe` 的常见 MoE 模块，对应策略为：

```python
ActivationRematConfig(
    save_regions=[
        "moe.routed_experts.w13.grouped_mm",
        "moe.routed_experts.w2.grouped_mm",
        "moe.routed_experts.token_dispatcher.dispatch",
        "moe.routed_experts.token_dispatcher.combine",
        "moe.shared_experts.*",
        "moe.tp_output_reduction",
        "moe.tp_routed_reduction",
    ]
)
```

没有 EP 时没有通信可保存，因此本地专家排序和分数加权 scatter-add 是普通操作。这些区域之外的操作，包括 token 分片零填充和分支加法，都会被重算。路由决策单独保留，以便重放期间专家选择保持一致。