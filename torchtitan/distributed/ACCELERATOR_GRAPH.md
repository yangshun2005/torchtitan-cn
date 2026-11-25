# 加速器图

加速器图捕获是可选的。加速器图记录一次 GPU 工作，然后用固定的张量地址重放。张量输入值可以在重放之间变化。输入形状和结构必须保持不变。

捕获会运行 Python 函数并记录其 GPU 工作，但不执行该 GPU 工作。包装器会立即重放一次图。因此，一次捕获调用算作一次真实运行。重放不会再次调用 Python 函数。

捕获之前，TorchTitan 会在不使用图的情况下运行工作以初始化状态。这些是预热调用。每次预热调用都是一次真实运行，其结果和状态变更会被保留。本文档用“eager”表示不使用加速器图的执行，用“eval”表示验证。一个训练步是一次优化器更新。

## 配置

一个微批次组是指不使用流水线并行（PP）时的一次本地前向/反向调用，或使用 PP 时的一次 PP 调度调用及其微批次。训练步中的所有组都贡献到同一次优化器更新。

训练捕获需要 `training.disable_accelerator_graphs=False`。

| 设置 | 捕获的工作 |
| --- | --- |
| `training.accelerator_graph_per_microbatch_group=False` | 一次优化器更新中所有微批次组的前向和反向 |
| `training.accelerator_graph_per_microbatch_group=True` | 一个微批次组的前向和反向，对每个组重放 |
| `optim.enable_accelerator_graph=True` | 梯度裁剪、有限性检查和优化器更新 |
| `training.disable_accelerator_graphs=False` | 一个验证微批次组（如果配置了验证） |

一个 eval 微批次组在不使用 PP 时包含一个微批次。使用 PP 时，它包含一次 eval 调度调用的微批次。一次完整验证运行会运行配置数量的组，并报告一个验证损失。

当 `accelerator_graph_per_microbatch_group=False` 时，一个图捕获所有微批次组。每次重放为一次更新运行所有组。组的数量及其输入结构必须在更新之间保持不变。

当 `accelerator_graph_per_microbatch_group=True` 时，一个图捕获一个微批次组。每个组重放该图。各组必须具有相同的输入结构、张量形状、dtype、设备和接收梯度的参数集合。组的数量可以在更新之间变化。每个组单独归约梯度。使用混合分片数据并行（HSDP）时，这可能改变结果，因为 eager 累积每次更新只归约一次副本。

训练和验证使用相同的捕获标志。禁用训练捕获也会禁用验证捕获。
优化器捕获需要启用训练捕获。学习率调度和指数移动平均（EMA）更新在优化器图之外运行。优化器捕获使用张量学习率，因此每次重放都读取当前学习率。

捕获需要 NVIDIA CUDA。捕获期间，CPU 不能等待 GPU 结果，即使在固定点也不行。这包括 `.item()` 和 `torch.cuda.synchronize()`。CPU 等待必须在捕获之外运行。GPU 流等待在 CUDA 捕获规则下是允许的。DistMoE 支持专家并行（EP）捕获。PP 需要支持所请求训练或 eval 工作的调度。FluxValidator、GraphTrainer 和 TorchFT 保持验证为 eager，并在启用训练捕获时发出警告。

## 共享池和捕获顺序

`AcceleratorGraphManager` 拥有一个内存池、一个捕获流和图包装器。该模块暴露一个实例 `accelerator_graph_manager`。训练、优化器和验证图共享其池。它们一次运行一个。

捕获期间，分配器可以在张量释放后复用其存储。这包括池中来自另一个图的临时张量。仍被活跃张量持有的存储不能用于捕获分配。

`AcceleratorGraphWrapper` 处理预热、捕获和重放。训练和优化器包装器各使用两次 eager 预热调用。第一次初始化状态，例如优化器张量。第二次在该状态就位的情况下运行。这是 TorchTitan 的预热策略，并非加速器图总是需要两次预热调用的要求。

eval 包装器使用一次 eager 预热调用。Eval 有单独的前向路径：模型使用 eval 模式，autograd 被禁用，PP 运行时不带反向。训练预热不会预热此路径。完整的首次验证运行保持 eager，即使它包含多个组。

启用 eval 捕获时，验证会阻止所有包装器的首次捕获。它们的预热调用仍在共享流上运行。验证在每次训练更新后运行，直到 eval 捕获。每次调用运行一次完整验证。

首次运行预热 eval。后续运行在所有已注册包装器完成预热后允许捕获。其第一个组捕获并重放 eval。其余组重放该图。训练和优化器可以在其下一次调用时捕获。后续验证遵循其配置的频率。

每次更新一次训练包装器调用且启用优化器捕获时：

```text
Step 1: Training warmup 1 -> optimizer warmup 1 -> eager eval
Step 2: Training warmup 2 -> optimizer warmup 2 -> eval capture/replay
Step 3: Training capture/replay -> optimizer capture/replay -> eval if due
```

预热计数的是包装器调用，不是训练步。使用按微批次组捕获时，一次更新中可能发生多次训练预热调用。捕获门控使后续组保持 eager，直到 eval 捕获。包装器在其首次调用时注册。`all_wrappers_warmed_up` 检查那些已注册的包装器。`capture_allowed` 控制没有图的包装器的捕获。它不会阻止现有图的重放。

首次进程本地验证运行在恢复运行中同样保持 eager。在捕获前结束的运行仍会报告其验证结果。禁用训练捕获时，验证遵循其配置的调度。

每次验证运行构建自己的加载器。不会丢弃任何预热结果。RNG 状态和模型缓冲区不会被恢复。如果 eval 使用 RNG 或更改模型缓冲区，启动验证可能改变后续训练结果。数值比较必须使用相同的验证调度。

返回之前，eval 释放模型输出和 PP 调度保存的张量。它还会完成完全分片数据并行（FSDP）清理。包装器只保留损失结果。

PP 调度在 eval 和训练之间切换时重建元数据。`prepare_pipeline_schedule_for_accelerator_graph()` 为所请求的模式准备元数据。它可以与其他 rank 交换元数据，不能在捕获内运行。包装器的 `prepare_for_capture` 回调在共享流上、捕获之外运行此设置一次。预热使用 eager 调度自身的初始化。

启用训练捕获时，在训练预热结束前捕获 eval 可能提高峰值内存。图池保持预留，而 eager 训练从普通池分配。等到预热结束可避免训练预热期间的这种重叠。共享流也避免了训练和 eval 的单独 eager 缓存。

每次捕获以 `torch.cuda.graph`（或 `torch.accelerator.Graph`）开始，它会同步设备并调用 `empty_cache()`。这会释放未使用的普通缓存。它不能释放活跃张量或活跃图池持有的存储。重放不会调用它。

## 使用和不使用 `_AcceleratorGraphGradientState` 时的内存使用

图示展示了共享池中一个 GPU 地址范围的可能复用。每一行显示其张量值何时需要。它们不显示总内存使用，也不承诺每个临时张量都共享梯度存储。

“临时”是指在下一次使用该地址范围之前被释放的张量。反向所需的激活必须保持活跃，直到反向消费它们。参数和优化器状态也保持活跃。优化器更新它们的值，但临时张量不能使用它们的存储。

图示使用一次前向/反向调用。使用多个微批次组或交错 PP 调用时，较早反向调用的梯度必须在后续前向调用期间保持完整。

### 不使用 `_AcceleratorGraphGradientState`

`zero_grad(set_to_none=False)` 保持 eager 预热期间分配的梯度缓冲区。这些缓冲区在图池之外。它们在前向、反向、优化器和 eval 期间保持分配。图分配器不能复用它们的存储。Eval 和训练仍可共享图池内的临时存储：

```text
Capture phase       eval            forward         backward        optimizer
Eager grad storage  [======================================================] ...
Eval temporary      [==========]
Forward temporary                   [==========]
Backward temporary                                  [==========]
Shared range        [eval temp]     [forward temp]  [backward temp]
```

重放使 eager 梯度缓冲区与临时存储保持分离：

```text
Replay phase        forward         backward        optimizer       eval
Eager grad storage  [======================================================] ...
Gradient values                     [========================]
Forward temporary   [==========]
Backward temporary                  [==========]
Eval temporary                                                      [==========]
Shared range        [forward temp]  [backward temp]                  [eval temp]
```

按微批次组捕获使用此方法。它每次更新用 `set_to_none=False` 清除梯度一次，并在组重放之间累积它们。

使用 `set_to_none=True` 而不保存和恢复梯度引用会失败。重放不会重复 autograd 对 `param.grad` 的赋值。该引用被清除后，优化器看到 `param.grad is None` 并跳过参数更新。

### 使用 `_AcceleratorGraphGradientState`

一起捕获所有微批次组使用 `_AcceleratorGraphGradientState` 保存和恢复梯度引用。训练捕获之前，`zero_grad(set_to_none=True)` 释放 eager 梯度缓冲区。首次前向开始时参数梯度不存在。Autograd 在图池内的反向期间创建它们。它可以使用前向张量释放的存储。

Eval 捕获先运行，并在训练捕获前释放其临时张量。这使同一地址范围可以服务于 eval、前向和梯度：

```text
Capture phase       eval            forward         backward        optimizer
Eval temporary      [==========]
Forward temporary                   [==========]
Gradient values                                     [======================]
Shared range        [eval temp]     [forward temp]  [gradient               ]
```

捕获后，状态保留对梯度张量的引用。每次训练重放之前，`zero_grad(set_to_none=True)` 清除 `param.grad`。保存的引用使存储保持活跃。重放在捕获的地址写入新梯度。然后状态恢复 `param.grad`，以便优化器可以读取它们。

捕获在分配梯度之前为前向临时张量选择了地址。重放使用这些地址而不向分配器请求新存储。当不需要梯度值时，前向内核可以将临时数据写入保存的梯度存储。Eval 也可以在优化器读取梯度后写入它，因为 eval 先被捕获：

```text
Replay phase        forward         backward        optimizer       eval           next forward
Gradient values                     [========================]
Forward temporary   [==========]                                                   [==========]
Eval temporary                                                      [==========]
Shared range        [forward temp]  [gradient               ]       [eval temp]    [forward temp]
```

梯度张量在“Gradient values”区间之外仍然存在。下一次反向写入新梯度。Eval 不得在反向和优化器更新之间运行。它不得与训练或优化器 GPU 工作重叠。

前向和 eval 可以共享临时存储。它们使用单独的图。训练图包含反向工作以及前向工作。Eval 使用 eval 模式并禁用 autograd。PP 和 FSDP 也可以使用不同的通信。

### 为什么 eval 先捕获

CUDA 允许在训练和优化器捕获之后捕获 eval。使用 `_AcceleratorGraphGradientState` 时，该顺序可能使用更多内存。状态保持梯度张量直到拆除。清除 `param.grad` 或完成优化器更新不会释放保存的引用。后续 eval 捕获无法在该活跃存储中分配张量。其他已释放的训练存储仍然可用。

当 eval 先捕获时，它在训练捕获分配梯度之前释放其临时张量。分配器可以将相同存储分配给那些梯度。重放期间，eval 和前向使用其记录的地址。当不需要梯度值时，它们可以写入该存储。

## 输入、输出和清理

`post_validate_fsdp_for_accelerator_graph()` 完成待处理的分组前向并重新分片完整参数，以便后续捕获可以复用其存储。使用分块损失时，模型跳过 `lm_head`，然后损失在根前向之外调用它。包含 `norm` 和 `lm_head` 的 FSDP 组仍可能有 `norm` 待处理。验证没有反向回调来清除该状态。Eager 和捕获验证使用相同的清理。

重放之前，包装器将张量输入值复制到用于捕获的缓冲区。标记为静态的输入不会被复制。它们的地址必须保持固定。非张量输入值以及张量形状、dtype 和设备也必须保持固定。`AcceleratorGraphInputSpec` 暴露存储在 `BlockMask` 中的张量，以便重放可以更新它们。

图输出指向图拥有的存储。在后续重放可能覆盖它们之前读取或克隆结果。共享池的另一个图也可能覆盖该存储。

`accelerator_graph_teardown()` 销毁所有已注册的图，并清除保存的输入、输出和梯度引用。`TrainingEngine.close()` 在拆除前完成待处理工作。梯度存储保持活跃，直到那些引用被释放。

对于内存检查，比较峰值分配内存、峰值保留内存和共享池大小。保留内存包括未使用的缓存。检查预热以及重放。CUDA 和通信库也使用 PyTorch 分配器之外的设备内存。