# 在 TorchTitan 中启用 TorchFT 容错

## 为什么在 TorchTitan 中使用 TorchFT？

TorchFT 旨在为使用副本权重进行训练的场景（如 DDP 或 HSDP）提供容错能力。在 TorchTitan 中启用 TorchFT 后，即使部分机器故障，训练过程也能继续。有关 TorchFT 的更多信息，请参阅 [TorchFT 仓库](https://github.com/pytorch/torchft/)。

**注意：** 这是一项仍在进行中的开发工作，所有内容均可能变更。

## 在 TorchTitan 中使用 TorchFT 的前提条件

在 TorchTitan 中使用 TorchFT 之前，需按照 [TorchFT README](https://github.com/pytorch/torchft/blob/main/README.md) 中的说明安装 TorchFT。

或者，也可以通过 `pip install torchft-nightly` 安装 TorchFT。

## 配置 TorchTitan 以使用 TorchFT

在 TorchTitan 中使用 TorchFT 时，需要启动多个副本组，每个副本组是一个独立的 TorchTitan 实例。每个副本组负责维护一份模型权重副本。发生故障时，其他副本组可以继续训练，不会丢失权重信息。

例如，要在单机八 GPU 上运行 HSDP，权重在四个 GPU 内分片，使用两个副本组（2, 4 设备网格），则在所选 recipe 中设置 `data_parallel_replicate_degree=2` 和 `data_parallel_shard_degree=4`。但要使用 TorchFT，需要启动两个 TorchTitan 实例，每个管理四个 GPU，并通过 TorchFT 相互通信。

## 示例配置

考虑一个示例：在单机八 GPU 上运行 HSDP，权重在四个 GPU 内分片，使用两个副本组（2, 4 设备网格）。不使用 TorchFT 时，在 `config.parallelism` 上设置相应字段。但一旦某个训练器故障（模拟真实机器故障），整个训练过程必须停止并从最后一个检查点恢复，这会导致显著的停机时间和资源浪费。

使用 TorchFT 后，可以容忍一个副本组故障，确保训练过程不中断。为此，启动两个 TorchTitan 实例，每个管理四个 GPU，并通过 TorchFT 相互通信。此设置可实现无缝容错，并将单个训练器故障的影响降到最低。
### 使用 TorchTitan 启动 TorchFT（示例 1）

要使用 TorchTitan 启动 TorchFT，需在不同 shell 会话中执行以下三条命令：

1. 启动 TorchFT lighthouse：

```bash
RUST_BACKTRACE=1 torchft_lighthouse --min_replicas 1 --quorum_tick_ms 100 --join_timeout_ms 10000
```

2. 启动第一个 TorchTitan 实例：

```bash
NGPU=4 CUDA_VISIBLE_DEVICES=0,1,2,3 MODULE=torchtitan_recipes.torchft.llama3 CONFIG=llama3_torchft_replica_0 ./run_train.sh
```
3. 启动第二个 TorchTitan 实例：

```bash
NGPU=4 CUDA_VISIBLE_DEVICES=4,5,6,7 MODULE=torchtitan_recipes.torchft.llama3 CONFIG=llama3_torchft_replica_1 ./run_train.sh
```

### 说明

* 由于在单机上运行，我们通过环境变量 `NGPU` 和 `CUDA_VISIBLE_DEVICES` 限制每个 TorchTitan 实例可见的 GPU。实际场景中，每个 TorchTitan 实例不会共享机器，因此无需这些变量。
* 各 recipe 中的 `fault_tolerance.enable` 启用 TorchFT 功能。
* `fault_tolerance.group_size=2` 告知 TorchTitan 有两个副本组。
* `fault_tolerance.replica_id` 标识每个实例。
* 注意，副本 ID 最小的存活副本组将执行检查点保存。

实际场景中，`torchft_lighthouse` 通常位于不同机器上。`TORCHFT_LIGHTHOUSE` 环境变量用于告知 TorchFT 如何与 `torchft_lighthouse` 通信，默认值为 `http://localhost:29510`。

### 使用半同步训练（示例 2）

TorchFT 提供的算法无需每步同步，副本组可以每 N 步同步一次权重。

**关于 Token 数量的说明**：对于 DiLoCo，token 数量术语有一个重要区别：

`training.num_tokens_per_train_step` 控制全局 token 预算。每次优化器步骤前累积的前向/反向迭代次数由该预算推导得出。

- **每步 Token 数**：所有 DiLoCo 岛/副本组的总 token 数
- **每步内部 Token 数**：每个 DiLoCo 岛内的 token 数，由每步 token 数除以副本组数量得出。

#### 副本组 0
```bash
MODULE=torchtitan_recipes.torchft.llama3 CONFIG=llama3_torchft_replica_0 CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 ./run_train.sh
```

#### 副本组 1
```bash
MODULE=torchtitan_recipes.torchft.llama3 CONFIG=llama3_torchft_replica_1 CUDA_VISIBLE_DEVICES=4,5,6,7 NGPU=4 ./run_train.sh
```

## 容错配置选项

在所选 Python recipe 中配置容错。

适用于所有训练模式：

- `config.fault_tolerance.use_async_quorum = False` 运行同步 quorum：
  该步骤在前向/反向之前等待 quorum，包括用于恢复的状态导出/加载。
  设置 `semi_sync_method` 时该字段被忽略，因为半同步训练始终使用同步 quorum。
  默认值（`True`）在步骤后台运行 quorum。

[可选] 仅适用于半同步训练：

- `config.fault_tolerance.sync_steps`：同步前的训练步数。
- `config.fault_tolerance.semi_sync_method`：同步方法（如 "local_sgd"、"diloco"）

更多半同步配置选项，请参阅 [config/job_config.py](config/job_config.py)。

### 超时配置

为使 TorchFT 正确恢复，以下超时必须构成严格的层级关系：
每一层必须长于其上一层，以便较低层超时先触发恢复，而非较高层超时杀死进程。

| 参数 | 推荐值 | 用途 |
| --- | --- | --- |
| `fault_tolerance.process_group_timeout_ms` | 60000（60s） | 最短超时——检测停滞的副本间集合通信（如因对端死亡而挂起的 allreduce）。TorchFT 捕获该超时并将该步骤标记为出错。必须短于 `comm.train_timeout_seconds`：TorchFT 的 FSDP all-reduce Hook 在 FSDP 操作内联运行，会阻塞默认进程组集合通信。 |
| `comm.train_timeout_seconds` | 120s | 稳态训练期间副本内 FSDP 集合通信的默认进程组超时。必须大于 `process_group_timeout_ms`，以便 TorchFT 进程组能在看门狗杀死进程前超时并恢复。 |
| `comm.init_timeout_seconds` | 180s | 同上，用于初始化，额外留出缓冲以应对惰性初始化、数据加载和 JIT 编译。 |
| `TORCH_NCCL_NONBLOCKING_TIMEOUT` | 200s | NCCL 内部 C++ 层非阻塞进程组操作的超时。必须大于 Python 层超时，以便 PyTorch 超时先触发，在 NCCL 于 C++ 层中止前进行优雅处理。 |
| `TORCHFT_TIMEOUT_SEC` | 180s | `ft.Manager` 操作（quorum、should_commit 等）的总截止时间。必须短于 `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC`。 |
| `TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC` | 900s | 必须大于以上所有超时，以免 TorchFT 仍在处理超时时 NCCL 杀死进程。通过环境变量设置。 |
| `TORCHFT_QUORUM_TIMEOUT_SEC` | 3600s | 最长超时——副本等待加入 quorum 的时间。必须覆盖节点供应时间。 |

注意：`TORCH_NCCL_NONBLOCKING_TIMEOUT` 只需大于其上的 Python 层超时，因此其相对于 `TORCHFT_TIMEOUT_SEC` 的顺序可以放宽。

## 环境变量

- `TORCHFT_LIGHTHOUSE`：lighthouse 服务的 URL
- `TORCHFT_MANAGER_PORT`：TorchFT manager 的端口
- `REPLICA_GROUP_ID`：副本组的标识符
- `RUST_LOGS`：Rust 组件的日志级别
- `RUST_BACKTRACE`：启用回溯以进行调试