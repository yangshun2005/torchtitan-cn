# TorchTitan 可观测性

分布式训练的结构化日志。按 rank 输出 JSONL 事件，用于阶段计时、诊断和事后分析。

设计原则：
- 对 LLM 友好：输出结构化的按 rank 数据，运行中和运行后都可查询。
- 同时支持 SPMD 预训练（单个进程组）和带多个独立 actor 的 RL（无共享进程组）。
- 对用户透明——无需传递指标字典。
- 绝不阻塞训练。
- 通过 handler 工厂支持可插拔后端。

## 快速开始

```python
from torchtitan.observability.logging import init_logger
from torchtitan.observability import structured_logger as sl

# console logger (stdout, [titan] prefix)
init_logger()

# Register handlers, i.e. the functions that will take the logs
# and save to a local jsonl, a database, or something else.
sl.init_structured_logger(source="training", output_dir="./outputs")

# Use to register a point-in-time marker — training has started.
sl.log_trace_instant("training_start")

loaded_step = 0
for step in range(loaded_step + 1, num_steps + 1):
    # Stamp every subsequent record with `step` and `relative_step`
    sl.set_step(step, relative_step=step - loaded_step)

    if should_garbage_collect:
        # Appends "gc" to `step_tags` on every record for this step; tags
        # reset at the next set_step() call.
        # Users can filter such steps later, e.g. "ignore if tag X".
        sl.add_step_tag("gc")
        with sl.log_trace_span("gc_collect"):
            run_gc()

    with sl.log_trace_span("fwd_bwd"):
        output = model(batch)
        loss.backward()

    with sl.log_trace_span("Optimizer"):
        optimizer.step()

    # Scalars you may want to register to help debug, for example.
    sl.log_trace_scalar({
        "num_trainable_tokens": num_trainable_tokens,
         "batch_size": bsz
         })
```

在任何 trace 调用之前，每个进程调用一次 `init_logger()` 和 `sl.init_structured_logger()`。rank 和 source 在初始化时写入 formatter；每条 JSONL 记录自动包含 `rank`、`source`、`caller`（file:line:function）、`time_us`、`step`、`relative_step`，以及（设置了标签时）`step_tags`。

## API 参考

完整参数见 docstring：

- `sl.init_structured_logger(source, output_dir, rank=None, enable=True)` —— 装配 handler；在任何 trace 调用之前每个进程调用一次。传 `enable=False`（或在 recipe 中设置 `debug.enable_structured_logging=False`）可让所有 trace 调用变为空操作。
- `sl.log_trace_span(event_type, description=None, *, stacklevel=2)` —— 上下文管理器 / 装饰器；输出 `_start` / `_end` / 可选的 `_error` 记录。
- `sl.log_trace_instant(event_type, *, stacklevel=2)` —— 时间点标记（无持续时间）。
- `sl.log_trace_scalar(scalars, *, stacklevel=2)` —— 从 `{name: number}` 字典输出 `metric_value` 记录。
- `sl.set_step(step, *, relative_step=None)` —— 为后续记录打上 step；清除上一步的标签。
- `sl.add_step_tag(tag)` / `sl.clear_step_tags()` —— 标注当前 step（如 `"gc"`、`"eval"`）。`set_step` 时会调用 `clear_step_tags`。
- `TITAN_STRUCT_LOGGER_HANDLERS` —— 在环境变量层面定义 handler

## 信息流

端到端来看，用户代码调用某个 `sl.` 辅助函数时会发生什么：

```
user code
    │   with sl.log_trace_span("fwd_bwd"):
    │       ...
    │
    │   # On entry, log_trace_span calls:
    │       _structured_logger.info(
    │           msg="[step 5] fwd_bwd_start",       ← registers human-readable string
    │           extra=event_extra(                  ← structured payload
    │               event_type="fwd_bwd_start",     ← → record.log_type_name
    │               step=5,                         ← → record.step
    │               task_name=None,                 ← → record.task_name
    │           ),
    │       )
    │
    │   # sl.set_step() / sl.add_step_tag() write into a ContextVar and a
    │   # module global; TraceJsonlFormatter reads them (get_step() /
    │   # get_step_tags()) when it formats the record on the calling thread,
    │   # so every record picks up the current step + tags.
    ▼
_structured_logger  (logging.Logger, name="torchtitan.structured_logger",
                     propagate=False — records stay out of the root logger)
    │
    ▼
TraceEventsOnlyFilter  (drops records that reached the logger WITHOUT a
                        log_type_name attribute — defensive; shouldn't fire
                        in practice because only the sl.* helpers write here)
    │
    ├── _JsonlQueueHandler     ──▶  TraceJsonlFormatter (calling thread)
    │        └──▶  queue  ──▶  listener thread  ──▶  TraceJsonlHandler  ──▶  {output_dir}/structured_logs/*.jsonl
    └── TraceMyDBHandler*     ──▶  TraceMyDBFormatter     ──▶  MyDB # extra handler defined by user
```

默认 JSONL handler 在调用线程上格式化每条记录，并在后台监听线程上写入，因此文件写入卡住（如 NFS 上）不会阻塞训练。解释器退出时，`logging.shutdown` 会写出所有排队记录；此后输出的记录（如来自较晚的 `atexit` 钩子或守护线程）会同步写入。进程被杀死时（SIGKILL、SIGTERM、原生崩溃、`os._exit`）仍在排队的记录会丢失；监听线程通常滞后远小于一毫秒，但调用线程用纯 Python 工作占住 GIL 时会更久。

## 自定义 handler

`TITAN_STRUCT_LOGGER_HANDLERS` 是逗号分隔的 Python 函数全限定路径列表。设置后，只运行列出的工厂。

```bash
export TITAN_STRUCT_LOGGER_HANDLERS="torchtitan.observability.structured_logger.jsonl_handler.register_jsonl_handler,mypackage.my_backend.register_my_db_handler"
```

handler 工厂接收 `sl.init_structured_logger()` 转发的参数，并向 `structured_logger` 附加一个 handler。示例：将事件流式发送到远程数据库而非写入磁盘，并在过程中为每条记录补充集群元数据。

```python
import logging
import os

from torchtitan.observability.structured_logger.jsonl_handler import (
    TraceJsonlFormatter,
)
from torchtitan.observability.structured_logger.structured_logging import (
    TraceEventsOnlyFilter,
)


class MyDBFormatter(TraceJsonlFormatter):
    """Enrich each record with backend-specific fields before serialization."""

    def _log_dict(self, record):
        d = super()._log_dict(record)
        d["cluster"] = os.environ.get("CLUSTER_NAME", "unknown")
        return d


class MyDBHandler(logging.Handler):
    """Send each trace event to a remote DB as it's emitted."""

    def __init__(self, rank: int, source: str, db_url: str):
        super().__init__()
        self.client = MyDBClient(db_url)
        self.setFormatter(MyDBFormatter(rank=rank, source=source))
        self.addFilter(TraceEventsOnlyFilter())

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.client.insert_row(self.format(record))
        except Exception:
            self.handleError(record)


def register_my_db_handler(
    *, structured_logger: logging.Logger, rank: int, source: str, **kw
) -> None:
    structured_logger.addHandler(
        MyDBHandler(rank=rank, source=source, db_url="mydb://..."),
    )
```

## 分析

### 从 JSONL 生成甘特图

```python
from torchtitan.observability.structured_logger.gantt_generator import (
    generate_gantt_trace,
)

generate_gantt_trace("outputs/structured_logs/", "outputs/analysis/gantt.json")
```

读取每个 rank 的 JSONL，配对 `_start` / `_end`，写出 Chrome Trace JSON。用 [Perfetto](https://ui.perfetto.dev) 打开。

示例 ASCII 草图：
```
                       step 1                              step 2
                  ┌──────────────────────────┐       ┌───────────────────────────┐
trainer rank 0    │ ▓▓ fwd_bwd ▓▓  ▓ optim ▓ │       │ ▓ fwd_bwd ▓▓ ▓▓ optim ▓   │
trainer rank 1    │  ▓▓ fwd_bwd ▓ ▓ optim ▓  │       │  ▓▓ fwd_bwd ▓▓ ▓ optim ▓  │
trainer rank 2    │ ▓ fwd_bwd ▓▓ ▓ optim ▓▓  │       │  ▓ fwd_bwd ▓▓▓ ▓ optim ▓▓ │
trainer rank 3    │  ▓▓ fwd_bwd ▓▓▓ ▓ optim ▓│       │ ▓ fwd_bwd ▓▓ ▓ optim ▓    │
                  └──────────────────────────┘       └───────────────────────────┘
controller        ▓▓▓▓▓▓ training_s ▓▓▓▓▓▓▓ ▓ scoring ▓ ▓▓▓▓▓▓ training_s ▓▓▓▓▓▓
rollouter         ▓▓▓                                     ▓▓▓
reward                                       ▓▓▓                              ▓▓▓
                  ├──────────────────────────┼───────────┼──────────────────────┤
                  0s                         3s          4s                     7s
```

每个 `log_trace_span` 变成一根条。每个源文件变成单独一行进程。自动命名的 asyncio 任务被紧凑排入连续的 `0..K-1` 区间，K 为峰值并发数，因此即使长时间异步运行（如 RL actor 派发大量并发 RPC），轨道数也保持可读。