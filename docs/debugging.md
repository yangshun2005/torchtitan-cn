## 调试配置值

要在不启动训练的情况下检查解析后的配置，使用 `--print-config`：

```bash
python -m torchtitan.train \
  --module torchtitan_recipes.models.llama3 \
  --config llama3_8b \
   --print-config
```

要列出支持的运行时 CLI 选项，运行 `python -m torchtitan.train --help`。训练行为在 recipe 中配置，而非通过通用 section 标志。参见[配置指南](../torchtitan/config/README.md)。

## 假后端调试

TorchTitan 有两种假进程组模式，因为它们回答不同的调试问题。它们有意不共享回退路径：

| 模式 | 物理进程数 | 真实通信 | 用于验证 |
| --- | --- | --- | --- |
| `fake` | 1 | 无 | 配置、逻辑 mesh 构建、rank 本地模型所有权、张量形状，以及 SPMD 坐标零处单个选定 PP rank 的 PyTorch 管理内存。 |
| `real_pp_fake_spmd` | 每个 PP rank 恰好 1 个 | 仅 PP 发送/接收 | 流水线调度、真实 PP 缓冲区和传输、加速器图捕获，以及 DP、TP、CP、EP 保持逻辑缩放时的 rank 本地内存。 |

当完整逻辑模型所需的 rank 数超过本地可用数量时，先使用纯 fake 模式。当问题涉及流水线传输或流水线缓冲区生命周期时，升级到 real-PP/fake-SPMD。SPMD 通信、数值和性能证据仍需真实分布式运行。

### 逻辑拓扑

`NGPU` 始终是逻辑 world size，不一定是启动的进程数。对于流水线度 `P`，每个流水线坐标包含 `NGPU / P` 个扁平化的 SPMD 坐标。这些调试模式始终表示 SPMD 坐标零，因此选定的逻辑全局 rank 为：

```text
logical_rank = pp_rank * (NGPU / P)
```

逻辑 world size 必须能被 `P` 整除。环境变量约定如下：

| 变量 | 纯 fake | Real PP / fake SPMD | 含义 |
| --- | --- | --- | --- |
| `NGPU` | 必需 | 必需 | 用于构建模型 mesh 的完整逻辑 world size。 |
| `FAKE_PP_RANK` | `P > 1` 时必需 | 无效 | 单进程所表示的逻辑 PP 坐标。 |
| `RANK` | 不使用 | 由 `torchrun` 设置 | 混合模式下的物理 rank 和 PP 坐标。 |
| `WORLD_SIZE` | 不使用 | 由 `torchrun` 设置 | 物理进程数，必须等于 `P`。 |
| `LOCAL_RANK` | 由 `run_train.sh` 设为 `0` | 由 `torchrun` 设置 | 进程的物理设备索引。 |
| `MASTER_ADDR`、`MASTER_PORT` | 不使用 | 由 `torchrun` 设置 | 真实 PP 组的标准 rendezvous 设置。 |
| `COMM_BACKEND` | `run_train.sh` 便捷变量 | 不要使用 | shell 启动器识别 `fake`；混合模式通过 `--comm-backend real_pp_fake_spmd` 选择。 |

### 纯 fake 示例

定义一个流水线和数据并行分片度均为 2 的 recipe。以下命令在单个物理 GPU 上以 PP2 构建该四 rank 作业的逻辑 rank `1 * 2 = 2`。它验证该 rank 的 stage、分片、准备好的权重、流水线元数据和内存所有权，而不创建 NCCL 进程组或传输对等数据。

```bash
NGPU=4 \
FAKE_PP_RANK=1 \
COMM_BACKEND=fake \
MODULE=my_debug_configs \
CONFIG=llama3_debugmodel_pp2_fsdp2 \
./run_train.sh
```

无 PP 时，省略 `FAKE_PP_RANK`；表示的 rank 为逻辑 rank 零。`run_train.sh` 默认将纯 fake 调用限制为一步训练，使该路径保持为诊断而非意外的基准测试。

### Real PP / fake SPMD 示例

使用相同的 PP2/FSDP2 recipe，以下命令为 PP2 启动两个物理进程。每个进程表示四 rank 逻辑作业中其 PP rank 的 SPMD 坐标零。`torchrun` 分配物理 rank 0 和 1；这些 rank 即 PP 坐标。TorchTitan 在它们之间创建一个真实 NCCL PP 组，并为其他所有轴创建假组。

```bash
NGPU=4 \
PYTORCH_ALLOC_CONF=expandable_segments:True \
torchrun \
  --nproc_per_node=2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  --role=rank \
  --tee=3 \
  -m torchtitan.train \
  --module my_debug_configs \
  --config llama3_debugmodel_pp2_fsdp2 \
  --comm-backend real_pp_fake_spmd
```

物理 world size 必须等于 PP 度。不要设置 `FAKE_PP_RANK`：物理 `RANK` 已提供该坐标。所有物理 rank 使用 SPMD 坐标零，因此通过逻辑 mesh 形成一条 PP 线。

### 内存调试工作流

保持模型、dtype、batch 几何、并行度、激活检查点、FSDP 策略和加速器图设置与目标真实作业一致。然后：

1. 选择正在调查其 SPMD 零所有权的逻辑 PP 坐标。
2. 在初始化后、完整优化器预热后、稳态前向/反向期间以及优化器完成后记录分配器摘要或快照。
3. 在候选配置之间比较相同的逻辑坐标和观测点。
4. 如果 PP 传输或缓冲区生命周期重要，用 real PP/fake SPMD 重新运行。
5. 当结论依赖通信、数值或性能时，以真实分布式运行收尾。

Fake 执行表示 PyTorch 管理的参数、优化器状态、准备好的量化权重、激活、梯度、流水线缓冲区，以及显式模型 arena（如 DistMoE scratch 和激活存储）。它不能忠实表示 NCCL 通信器分配、网络注册、集合通信 scratch、SPMD 集合通信延迟或通信重叠。

### 解读失败

- 整除性或坐标错误是启动约定失败。修正逻辑拓扑，而不是更改模型形状来绕过它。
- 混合 world-size 错误意味着每个 PP rank 没有恰好一个物理进程。
- 纯 fake 成功但混合模式失败，将问题隔离到 PP 传输、PP 缓冲区所有权、通信器初始化或仅由真实流水线通信触发的其他路径。
- 混合模式成功但真实运行失败，指向真实 SPMD 集合通信、通信内存或依赖规模的调度行为。
- 任一 fake 模式成功都不能证明 loss 等价、分布式正确性、吞吐量或通信重叠。

[fake 分布式后端 skill](../.claude/skills/fake_distributed_backend/SKILL.md) 包含用于可重复内存调查的操作检查清单。

## 分布式断点和 LOG_RANK

`run_train.sh` 默认将 `LOG_RANK` 设为 `0`，并作为 `--local-ranks-filter` 传给 `torchrun`，因此只有 rank 0 的 stdin/stdout 被 tee 到控制台。因此，在被过滤的 rank 上调用 `torch.distributed.breakpoint(rank=N)` 会挂起且永不打印提示符。

要调试 rank N，在启动前将 `LOG_RANK` 设为 N（或包含 N 的逗号分隔列表）。不要更改 `run_train.sh` 中的默认 `LOG_RANK`。

```bash
LOG_RANK=1 ./run_train.sh
# 或同时保持 rank 0 可见：
LOG_RANK=0,1 ./run_train.sh
```

## 排查超时作业

如果遇到超时作业，需要调试以确定根因。为此，我们启用了 Flight Recorder，一个持续收集作业诊断信息的工具。
作业超时时，Flight Recorder 会自动在每个 rank 上生成包含有价值调试数据的 dump 文件。可在 `dump_folder` 目录中找到这些 dump 文件。
要了解如何使用这些日志分析和诊断问题，请参阅分步教程[链接](https://pytorch.org/tutorials/prototype/flight_recorder_tutorial.html)。



## 运行间的可复现性

调试多维并行（FSDP、TP、PP、CP、EP 的组合）问题时，确保可复现行为对隔离和修复问题至关重要。`torchtitan` 提供多种机制实现确定性训练运行。有关确保可复现性和管理 PyTorch 随机性的更多信息，可参阅 PyTorch 官方随机性文档：[PyTorch Randomness Documentation](https://docs.pytorch.org/docs/stable/notes/randomness.html)。

### 种子配置
在选定的 recipe 中设置一致的随机种子：

```python
config.debug.seed = 42
```

**并行下的种子行为：**
- **数据并行（DP/FSDP）、张量并行（TP）、上下文并行（CP）：** 所有 rank 使用相同种子。
    - 注意：对于 FSDP 和 TP，DTensor 会做特殊 RNG 管理，确保 Replicate 张量在各 rank 间获得相同初始化，但 Shard 张量在各 rank 间获得"随机"式初始化。
- **流水线并行（PP）：** 每个 PP stage 获得不同种子，确保不同 PP rank 上各层初始化不同。


### 确定性模式

在 recipe 中启用确定性算法，确保运行间逐位可复现：

```python
config.debug.deterministic = True
```

**作用：**
- 强制所有 CUDA 操作使用确定性算法
- 禁用 CuDNN 基准测试并启用确定性模式
- 为 CuBLAS 操作设置确定性工作区配置
- **注意：** 这会显著降低训练性能，但确保精确可复现

`debug.deterministic_warn_only` 仅用于探索性调试，不得用于数值验证。

### 基于种子检查点的可复现性

对于使用不同并行配置的多次实验运行，需要使用"种子"检查点确保各运行间模型初始化相同。这是因为训练引擎先分片模型参数，然后在每个 rank 上分别初始化权重。因此，在一个 rank 上初始化模型再分片并不等价。使用种子检查点可让不同运行从检查点加载相同模型权重——DCP 重分片会确保加载的权重按并行配置正确分片。

#### 创建种子检查点

创建注册配置，设置 `create_seed_checkpoint=True`、非 `None` 的 `checkpointer`，并将所有并行度设为 1，然后在单个设备上运行。

#### 加载种子检查点用于调试

使用种子检查点进行调试或验证时，可启用 `load_only` 配置，在训练期间加载检查点而不保存新检查点。这在你只想验证模型正确性或比较不同配置而不弄乱磁盘时特别有用：

在配置注册中设置 `checkpointer=CheckpointManager.Config(load_only=True)`。`load_only` 设置阻止训练过程保存任何检查点，允许你：
- 运行调试会话而不生成不需要的检查点文件
- 使用相同初始权重比较模型行为，无检查点开销

**注意**：使用种子检查点只能确保配置变化时模型具有相同初始权重，但即使设置了种子和 `deterministic` 模式，训练过程也可能不同，例如由于张量形状变化、数据精度变化、模型代码中使用随机性等。

## 验证分布式训练技术

新的并行或训练技术不应改变模型所学内容。用以下两种方式之一对照可信基线验证。

### 保持数据一致

两种比较都需要每次运行以相同顺序消费相同数据，因此保持每个优化器步的 token 数固定：

```
training.num_tokens_per_microbatch_per_dp_rank
* parallelism.num_pp_microbatches   # PP 禁用时为 1
* 数据并行度              # dp_replicate * dp_shard
* 推导的梯度累积步数
```

同时保持数据并行度固定，因为它决定 dataloader 如何分片数据集。例如，总 DP 度为 4 时：

- 运行 1：dp_shard = 4
- 运行 2：dp_replicate = 2, dp_shard = 2, TP = 2
- 运行 3：dp_replicate = 2, dp_shard = 2, CP = 2, PP = 2

某些并行会改变微批处理：PP 需要每个 stage 足够多的微批以保持气泡小。通过用每微批 token 数换取 PP 微批数来保持上述乘积固定，例如 1D FSDP 每微批 32768 token，而 FSDP + PP 用四个 PP 微批时为 8192 token。

### 逐位比较

当技术不应改变计算时（例如重构或激活检查点），各运行应产生逐位相同的 loss 和梯度范数。在每次运行中加载相同种子检查点，启用 `config.debug.seed` 和 `config.debug.deterministic`，并用 `scripts/loss_compare.py` 比较全精度 TensorBoard 值（stdout 四舍五入到五位有效数字）。

### 收敛比较

当不期望精确一致时（不同并行布局改变累积顺序；除非使用种子检查点，种子初始化在不同分片布局间不同），在代表性数据集（如 C4）上比较一次训练运行的 loss 曲线：

- **对照组：** 在 `dp` 个 GPU 上用可信 recipe 的 1D FSDP 作业，或其他已验证设置。即使测试更高维设置的优化（如异步 TP），也使用 1D FSDP：小的可接受差异可能在传递比较链中累积。
- **被测并行：** 至少在 `dp*p` 个 GPU 上、`p >= 2` 的 2D 作业。某些并行需要更大的 `p` 才能暴露数值问题；例如 CP 在默认混合精度下以 bf16 累积中间结果。对于相互兼容的并行，在高维组合中收敛意味着在其低维子集中收敛，因此一个 N-D 测试作业可覆盖多种并行。
- **被测其他技术：** 在其目标并行之上启用该技术的作业，例如异步 TP 用 FSDP + TP + `torch.compile`，或新 PP 调度用 FSDP + PP。

[Llama 3 README](../torchtitan/models/llama3/README.md#loss-convergence) 展示了比较 1D FSDP 与 3D 和 4D 作业的参考运行。