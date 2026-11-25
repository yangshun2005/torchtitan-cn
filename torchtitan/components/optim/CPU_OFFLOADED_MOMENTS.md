## CPU 卸载的 Adam 动量

将 Adam/AdamW 配置的 `moment_cpu_offload_fraction` 设为某个比例，即可把该比例的本地 `exp_avg` 和 `exp_avg_sq` 字节保留在锁页 CPU 内存中，而非加速器内存。`0.0`（默认值）将所有动量保留在设备上。`1.0` 卸载全部动量。未改动的融合 Adam 内核通过 CPU-GPU 链路原地读写卸载的动量。没有拷贝、旁路流或额外内核，该步骤仍可被 CUDA 图捕获。

```python
from dataclasses import replace

optimizers = config.optim.optimizer.optimizers
optimizers[0] = replace(optimizers[0], moment_cpu_offload_fraction=0.5)
```

`replace` 会执行下方的配置检查。直接赋值该字段会跳过这些检查。

单位是整个参数。优化器按顺序遍历参数，逐个卸载每个参数的动量，直到卸载字节数达到全部本地动量字节的该比例。其余动量保留在设备上。FSDP 均匀分片且专家大小相等，因此所有 rank 通常选择相同的参数。卸载的动量和设备上的动量共享一个参数组和一次融合 AdamW 调用。

### 分配器设置

默认情况下，PyTorch 锁页分配器会将每次分配向上取整到 2 的幂，这可能使动量的主机内存翻倍。卸载动量时，请对大块分配关闭取整：

```bash
PYTORCH_ALLOC_CONF="expandable_segments:True,pinned_max_round_threshold_mb:1" ./run_train.sh
```

如果你已设置 `PYTORCH_ALLOC_CONF`，请将 `pinned_max_round_threshold_mb:1` 追加到你的值后面。`run_train.sh` 会保留你的值，仅在未设置时使用 `expandable_segments:True`。使用此设置后，超过 1 MB 的分配将按其精确大小锁定，而非向上取整到 2 的幂。较小的分配仍会取整。超过 `pinned_max_cached_size_mb`（若已设置）的分配永不取整。分配动量后，优化器会将 `torch.cuda.host_memory_stats()` 报告的锁页字节数与请求的字节数进行比较。如果取整浪费超过 1 GiB，则抛出 `ValueError`。

### NUMA 绑定

`TrainingEngine`（`torchtitan/training_engine.py`）调用 `maybe_apply_numa_binding`，在任何锁页内存分配之前将每个 rank 绑定到其 GPU 的 NUMA 节点，因此动量落在 GPU 的节点上。若不绑定，在 GB300 上四块 GPU 繁忙时，双向同时拷贝下降约 2-2.5 倍（86-111 GB/s 降至 44 GB/s）。仅主机到设备拷贝下降 1.0-1.8 倍。

### 选择比例

使用能容纳的最小比例，因为每个卸载字节都会增加步骤耗时。例如，逐步提高该比例，直到首次优化器步骤即内存不足的运行能够容纳。该功能面向具有快速 CPU-GPU 链路的主机，如 GB300。尚未在 PCIe 主机上测量，在那些主机上步骤会慢得多。

激活卸载（torchtitan#6020）是互补的：它移动激活，而非优化器状态。两者使用相同的锁页分配器和 NUMA 绑定。

### 开销

每个优化器步骤通过主机链路对每个卸载动量读写一次：

```
每步字节数 = 2（读 + 写）x 2 个动量 x 动量字节数 x 卸载的参数
           = fp32 动量下每个卸载参数 16 字节
```

AdamW 时间增加约等于这些字节数除以主机链路带宽。每个卸载参数，加速器内存减少、锁页主机内存增加 `2 x 动量字节数`。在 GB300 上，预计每 GiB 卸载动量每步约 13.5 ms。

在一块绑定到其 NUMA 节点的 GB300 GPU 上，64-GPU 伪进程组 Muse Spark 运行（DP8/CP8/EP64，32K 序列长度，全部动量卸载）每次 AdamW 步骤移动 2 x 104 GB。在一次同步测量中，仅 AdamW 就耗时约 0.63 s（约 330 GB/s），占步骤的 5.8%。一个 Grace 插槽的两块 GPU 共享其主机带宽，因此当两者同时运行该步骤时，AdamW 耗时约 2 倍。

### 工作原理

优化器在构造时为卸载的参数创建状态。每个动量是一个来自 `torch.zeros(..., pin_memory=True)` 的 CPU 张量，以别名该内存的张量形式暴露给 CUDA（`__cuda_array_interface__`）。DTensor 参数获得具有相同 mesh 和 placements 的 DTensor 动量。其他参数在首次步骤时惰性获得设备动量。

```
GPU: param, grad --fused Adam--> param
                     ^    |
                     |    v  host link, in place
CPU (pinned):  exp_avg, exp_avg_sq
```

DCP 像保存和加载设备动量一样保存和加载卸载的动量，因此检查点可在任意两个比例之间互换。加载会写入相同的锁页缓冲区，无需在加速器内存中临时拷贝。

`optimizer.state_dict()` 将每个卸载动量作为其锁页内存上的 CPU 张量返回，因此检查点工具在 CPU 上拷贝它。两个主机缓冲区之间的 CUDA 拷贝是同步的，且慢得多。`optimizer.load_state_dict()` 若加载的值是其他张量，则将其拷贝到该内存中，并将 CUDA 动量保留在优化器状态中。

### 要求

当比例大于 0 时，以下情况会以 `ValueError` 拒绝：

- `fused=False`（配置时）。其他 Adam 实现会多次遍历动量。
- CUDA 以外的加速器，包括 ROCm（配置时）。
- 卸载的参数不在 CUDA 上（优化器构造时）。这包括 `parallelism.fsdp_enable_cpu_offloading` 和 `create_seed_checkpoint` 运行，它们需要 `moment_cpu_offload_fraction=0`。种子检查点仅含模型，因此训练运行仍可卸载。
- 卸载参数的非连续本地分片（优化器构造时）。
- 锁页内存取整超过 1 GiB（优化器构造时）。参见[分配器设置](#allocator-setting)。
- TorchFT（配置时）。副本恢复会用设备张量替换锁页动量。

比例超出 [0, 1] 也会在配置时被拒绝。