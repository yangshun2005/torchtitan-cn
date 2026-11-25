# Distributed

本包包含 PyTorch 原生并行机制，以及必须与之组合使用的训练技术。并非每个模块都会发起集合通信：激活重计算、加速器图和本地编译按设备运行，但它们放在这里，因为每个模块都与分片、流水线调度或 SPMD 区域交互，并在模型并行化期间应用。

## 并行机制与通信

| 模块 | 用途 |
| --- | --- |
| `parallelism_context.py` | 设备网格构建及每种并行机制的命名网格轴。见 [PARALLELISM_CONTEXT.md](PARALLELISM_CONTEXT.md)。 |
| `spmd_types.py` | 张量并行及其他分片区域的 SPMD 类型注解。 |
| `fsdp.py` | FSDP2 / HSDP 应用。见 [FSDP.md](FSDP.md)。 |
| `pipeline_parallel.py` | 流水线阶段与调度。见 [PIPELINE_PARALLEL.md](PIPELINE_PARALLEL.md)。 |
| `context_parallel.py` | 上下文并行序列分片。 |
| `deepep/` | DeepEP / HybridEP 专家并行 token 分发。 |
| `flex_shard/` | 与参数存储布局不同的优化器计算布局（DistMuon）。见 [flex_shard/README.md](flex_shard/README.md)。 |
| `utils.py` | 进程组初始化、分布式归约、确定性、梯度裁剪。 |

## 与并行机制组合使用的技术

| 模块 | 用途 |
| --- | --- |
| `activation_remat.py` | 基于 `torch_remat` 的激活重计算（激活检查点）配置。见 [ACTIVATION_REMAT.md](ACTIVATION_REMAT.md)。 |
| `activation_offload.py` | 将激活重计算为反向传播保留的激活卸载到 CPU。见 [ACTIVATION_OFFLOAD.md](ACTIVATION_OFFLOAD.md)。 |
| `accelerator_graph.py` | 训练步骤的加速器图捕获与重放。 |
| `local_compile.py` | 对命名模型区域执行 `torch.compile`。 |
| `batch_invariant.py` | 批次不变（位级可复现）内核。 |
| `offloading.py` | 将每个 rank 的 NUMA 绑定到其 GPU 所在 CPU 节点。 |

## 首次优化器更新

`Optim.step()` 在首次调用时、梯度裁剪之前释放未使用的 CUDA 缓存。每个新的 `Optim` 实例各自跟踪自己的首次调用，因此恢复的步数不会跳过释放。释放前会等待 GPU 工作完成。这两次调用都在优化器加速器图捕获之外运行。

NCCL 在 PyTorch 缓存分配器之外分配其内部通信缓冲区。未使用的 PyTorch 缓存可能阻塞这些分配，即使释放缓存后活跃张量和 NCCL 缓冲区能够容纳。见 [issue #5019](https://github.com/pytorch/torchtitan/issues/5019)。

此变通方案要求：

- 前向和反向传播成功完成。
- 有足够的未使用 CUDA 缓存可释放给 NCCL 缓冲区。
- 受影响的 NCCL 分配发生在首次优化器调用期间。

释放会保留活跃张量和活跃的加速器图池。它不会减小 NCCL 缓冲区大小，也不会防止前向、反向传播期间或运行后期出现的分配失败。它不会释放 NCCL 主机共享内存。

## 环境设置

**`TORCH_NCCL_AVOID_RECORD_STREAMS=1` 配合张量并行。** TP 发起异步集合通信以重叠通信与计算。默认情况下，进程组会对输入和输出调用 [`Tensor.record_stream`](https://pytorch.org/docs/stable/generated/torch.Tensor.record_stream.html)，这将缓存分配器的内存复用绑定到 GPU 内核时序：集合通信运行期间，后续分配无法复用其张量的内存，即使需要这些张量的操作排在其后，于是内存堆积。设置此变量后，进程组改为持有张量引用，直到调用 `wait()`。只有在从不调用 `wait()` 时才会导致内存回退，而这并非预期用法。

**`CUDA_DEVICE_MAX_CONNECTIONS` 配合 FSDP2 和原生 TP。** 将其设为 CUDA 流数量（例如 16 或 32），以便计算和 NCCL 内核重叠。这与 Megatron 风格的 TP 相反，后者通常要求 `CUDA_DEVICE_MAX_CONNECTIONS=1`；不要照搬该设置。