# 性能分析

## 启用内存分析

在选定的 recipe 中配置性能分析：

```python
from torchtitan.observability.profiler import MemorySnapshotProfiler
from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel


def my_debug_recipe() -> Trainer.Config:
    config = llama3_debugmodel()
    config.profiler.memory_snapshot = MemorySnapshotProfiler.Config(
        save_folder="memory_snapshot", freq=3
    )
    return config
```

用 `MODULE=my_project.recipes CONFIG=my_debug_recipe ./run_train.sh` 启动。若发生 OOM，快照写入 `step_{step:012d}_exit`；常规快照使用 `step_{step:012d}`。

* `save_folder`：配置输出文件夹。
* `freq`：控制常规快照的采集频率（默认 10）。
* `max_entries`：限制记录的分配/释放历史条数。

保存的 pickle 文件位于输出文件夹中。要可视化快照文件，可将其拖放到 <https://pytorch.org/memory_viz>。内存分析的更多细节请参阅此[教程](https://pytorch.org/blog/understanding-gpu-memory-1/)。

## 使用 Cuspy 分析 CUDA

将 `profiler.trace` 设为 `CuspyTraceProfiler`（而非 `KinetoTraceProfiler`），即可用 Cuspy（PyTorch 的进程内 CUPTI 采集器）代替 Kineto 采集 CUDA 活动。在 Kineto 记录内容之外，它还能补充：

* 通过 CUPTI PM 采样获取的 GPU 性能指标，例如 SM 活动、DRAM 和 NVLink 带宽。
* GPU 功耗、时钟和温度计数器（`enable_environment_counters`）。
* CUDA 图依赖边和事件节点 ID。
* Perfetto protobuf 输出（`trace_format="pftrace"`），比 JSON 更小。

底层选项见 [`torch.profiler.CuspyConfig`](https://docs.pytorch.org/docs/main/profiler.html#torch.profiler.CuspyConfig)。

要求：

* 2026-09-21 或之后的 PyTorch nightly（或 PyTorch 2.15 及更新版本发布后）。
* 支持 CUDA 13 的 NVIDIA 驱动（R580 或更新）。
* libcupti 13.3 或更新（`pip install cupti-python`；CUDA 13.4 的 PyTorch wheel 已内置）。

若 torch 或 libcupti 不支持 Cuspy，性能分析器启动时会抛出 `ValueError`。

在 Python recipe 中启用：

```python
from torchtitan_recipes.tests.models.llama3 import llama3_debugmodel
from torchtitan.observability.profiler import CuspyTraceProfiler


def llama3_debugmodel_cuspy():
    config = llama3_debugmodel()
    config.profiler.trace = CuspyTraceProfiler.Config(
        performance_metrics=["sm__cycles_active.avg"]
    )
    return config
```

启用 CUDA 图时，前向-反向、优化器和验证图会以基于 CUPTI 的注解捕获，因此重放的 kernel 保留其 `mark_kernels` 元数据。这会在首次捕获时启动 Cuspy 的 CUPTI 订阅，并在本次运行的剩余时间内保持，因此其他 CUPTI 工具（Nsight Systems、通过 dynolog 按需触发的 Kineto trace）无法附加到该进程。

设置 `record_py_stacks=True` 可同时记录每个 CUDA 图 kernel 的 Python 启动栈。栈会与每个 trace 一同写入 `rank{rank}_cuda_graph_py_stacks.json.gz`，以 trace 中 kernel 事件的 `(graph id << 32) | graph node id` 为键。CUDA 图之外启动的 kernel 没有栈。

### PM 采样配置

`performance_metrics` 接受 CUPTI 指标名，例如 `sm__cycles_active.avg` 或 `dram__throughput.avg.pct_of_peak_sustained_elapsed`，每个指标在 trace 中成为一个计数器轨道。整个指标集必须能在单次硬件采集中采集。

要查看当前 GPU 支持哪些指标，可列出其基础指标名。完整指标名由基础名加后缀构成，如 `.avg` 或 `.avg.pct_of_peak_sustained_elapsed`：

```python
from torch.profiler._cuspy.pm_sampling import supported_metrics

print(sorted(supported_metrics()))  # e.g. "sm__cycles_active" -> "sm__cycles_active.avg"
```

* `pm_sampling_interval_ms`：采样间隔。默认 1 ms。
* `pm_lookback_window_ms`：缓冲区保留的采样历史时长。默认 10 s。

性能分析结束时，trace 会包含缓冲区中仍保留的 PM 采样。若预热和活跃步骤的总时长超过回看窗口，最早的采样会被丢弃。

采样缓冲区在分析期间保存在进程内存中，不写入 trace。CUPTI 为每个采样存储原始硬件计数器（每个数 KB，取决于指标），因此其大小随回看窗口/间隔增长：例如 100 s / 2 ms、7 个指标时约 1.1 GiB。