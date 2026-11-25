# TorchTitan RL 指标

Actor 和控制器发出带类型的 `Metric(key, value)` 记录。聚合在 `MetricsProcessor.log` 调用时惰性完成。日志记录器每步聚合一次这些记录，并把扁平的 `dict[str, float]` 发送到控制台和后端日志记录器。

vLLM 引擎级指标使用独立的 OpenTelemetry 流水线，详见 [vLLM 引擎指标](#vllm-引擎指标)。

```text
loss / trainer / controller
        |
        v
metrics = list[Metric(key, MetricValue)]
        |
        v
MetricsProcessor.log(step, metrics) -> MetricsProcessor._aggregate_metrics(metrics)
        |
        v
console + W&B / TensorBoard backends
```

## 用法

```python
from torchtitan.rl.observability import metrics as m

# The allow list is used to only log to console some keys, instead of all of them
config.metrics.console_log_keys_train = ["loss/mean", "rollout/prompt_length/mean"]
metrics_processor = config.metrics.build(
    log_dir=config.dump_folder,
    job_config=config.to_dict(),
)

metrics = []

# Add to a list multiple records with the same key
# They will be later reduced to: {"rollout/prompt_length/mean": 200}
for prompt_length in [100,300]:
    metrics.append(
        m.Metric("rollout/prompt_length", m.Mean(prompt_length)),
    )

# `from_list` is preferred when observations are already in a list
# since it creates a single record, instead of N.
metrics.append(m.Metric("reward", m.SummaryStats.from_list([0.0, 1.0])))

# You can define multiples key with different value types.
# Their final names have a different suffix `key/max` and `key/mean`.
response_lengths = [1000, 2000]
metrics += [
        m.Metric("rollout/response_length", m.Max.from_list(response_lengths)),
        m.Metric("rollout/response_length", m.Mean.from_list(response_lengths))
    ]

# Your train already returns reduced metrics
# You can use NoReduce as a no-op metric type.
metrics.append(m.Metric("loss/mean", m.NoReduce(0.42)))

# Log the metrics at step 7.
metrics_processor.log(step=7, metrics=metrics, is_validation=False)
```

在 log 调用时，它会聚合所有指标，生成如下字典：
```python
{
    "rollout/prompt_length/mean": 200.0,        # mean of [100, 300]
    "reward/_max": 1.0,                         # SummaryStats expansion
    "reward/_mean": 0.5,
    "reward/_min": 0.0,
    "reward/_std": 0.5,
    "reward/_sum": 1.0,
    "rollout/response_length/max": 2000.0,      # max of [1000, 2000]
    "rollout/response_length/mean": 1500.0,     # mean of [1000, 2000]
    "loss/mean": 0.42,                          # NoReduce pass-through
}
```

完整字典会转发给每个后端（W&B、TensorBoard）。控制台输出按配置的允许列表过滤。使用上面的允许列表（`["rollout/prompt_length/mean", "loss/mean"]`），打印行为：
```text
----------
Train | Step:  7  loss/mean: 0.42   rollout/prompt_length/mean: 200.0
```

## 指标值

```text
Mean
  Metric("reward", Mean.from_list([1.0, 3.0]))
  -> reward/mean = 2.0

Max
  Metric("reward", Max.from_list([1.0, 3.0]))
  -> reward/max = 3.0

Min
  Metric("reward", Min.from_list([1.0, 3.0]))
  -> reward/min = 1.0

Sum
  Metric("reward", Sum.from_list([1.0, 3.0]))
  -> reward/sum = 4.0

Std
  Metric("reward", Std.from_list([1.0, 3.0]))
  -> reward/std = 1.0

SummaryStats
  Metric("reward", SummaryStats.from_list([1.0, 3.0]))
  -> reward/_max, reward/_mean, reward/_min, reward/_std, reward/_sum

NoReduce
  Metric("loss/mean", NoReduce(0.42))
  -> loss/mean = 0.42
```

`SummaryStats` 使用带前导下划线的输出名，避免其子键与同一键下裸的 `Mean`/`Max`/`Min`/`Std`/`Sum` 记录冲突。

## 控制台输出

`MetricsProcessor.log(...)` 读取配置的允许列表：

```python
m.MetricsProcessor.Config(
    console_log_keys_train=["loss", "grad_norm"],
    console_log_keys_validation=["loss"],
)

metrics_processor.log(step=step, metrics=train_metrics)
# prints "Train | Step:  N | loss:  0.42 | grad_norm:  0.01

metrics_processor.log(step=step, metrics=val_metrics, is_validation=True)
# prints "Validation | Step:  N | loss:  0.42"
```

## 后端

`MetricsProcessor.Config.loggers` 按配置类型选择导出器。当该列表非空时，需向 `MetricsProcessor.Config.build(...)` 传入 `log_dir`。

```python
from torchtitan.observability.metrics import TensorBoardLogger, WandBLogger

config.metrics.loggers = [
    WandBLogger.Config(),
    TensorBoardLogger.Config(),
]
```

`WANDB_PROJECT` 选择 W&B 项目；监督学习和 RL 运行的默认值均为 `torchtitan`。日志记录器构造不会改变环境。自定义导出器参见[指标导出器配置](../../../observability/METRICS.md#configure-metric-exporters)。

## vLLM 引擎指标

`VllmOtelStatLogger` 通过 [OpenTelemetry](https://opentelemetry.io/) 导出引擎级 rollout 指标。该日志记录器默认禁用，仅当 `generator.engine.stat_logger` 包含 `VllmOtelStatLogger.Config` 时才注册。启用后，`OTEL_METRICS_EXPORTER` 必须选择 `jsonl` 或 `otlp`。这一显式门控可防止继承的 OpenTelemetry 环境变量意外激活日志记录器。每个 DP 副本中只有 TP rank 0 创建日志记录器，因为所有 TP rank 看到相同的引擎聚合统计。

导出的指标按 OpenTelemetry 仪器类型分组：

- 计数器：`vllm.generation_tokens`、`vllm.prompt_tokens`、`vllm.cached_prompt_tokens`、`vllm.preempted_requests`、`vllm.finished_requests`、`vllm.prefix_cache_queries` 和 `vllm.prefix_cache_hits`。
- 直方图：`vllm.time_to_first_token`、`vllm.inter_token_latency`、`vllm.decode_time`、`vllm.queue_time` 和 `vllm.e2e_latency`。
- 仪表：`vllm.kv_cache_usage`、`vllm.num_running_requests` 和 `vllm.num_waiting_requests`。

计数器是累积的；在后端中根据其速率推导吞吐量和前缀缓存命中率。直方图值以毫秒为单位。仪表报告日志记录器观察到的最新调度器状态。

#### 前缀缓存命中率

日志记录器将前缀缓存命中和查询导出为独立的单调计数器。根据其速率计算所选查询窗口内的命中率：

```text
prefix_cache_hit_rate =
    rate(vllm.prefix_cache_hits) / rate(vllm.prefix_cache_queries)
```

将结果乘以 100 以显示百分比。查询多个生成器时，先对各计数器的速率求和再相除：

```text
prefix_cache_hit_rate_percent =
    100 * sum(rate(vllm.prefix_cache_hits))
        / sum(rate(vllm.prefix_cache_queries))
```

查询速率为零时不返回值。使用速率而非原始累积计数器值，可将结果限制在所选时间窗口内，并处理计数器重置。

### 选择合适的导出器选项

两个导出器使用相同的 OpenTelemetry 仪器和指标名；仅目标不同：

- 本地开发、调试以及没有 OpenTelemetry collector 的短时或小规模运行时，使用 `jsonl`。
- 需要集中聚合、保留、仪表板或告警的分布式或生产运行时，使用 `otlp`。

#### 本地写入 JSONL

完成 [DAPO Math 设置](../../examples/dapo_math/README.md#setup)后，使用 `jsonl` 在无 collector 的情况下本地检查：

如果未配置 W&B，在所选 recipe 中设置 `config.metrics.loggers = []`。

```bash
OTEL_METRICS_EXPORTER=jsonl \
VLLM_LOG_STATS_INTERVAL=10 \
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.dapo_math \
  --config rl_dapo_qwen3_4b_math_8k
```

每个 TP rank 0 生成器将紧凑 JSON 记录写入 `{dump_folder}/vllm_metrics/{generator_name}.rank{rank}.jsonl`。此模式需要配置输出目录。

#### 使用 OTLP 导出

使用 `otlp` 将指标直接发送到 OpenTelemetry collector 或兼容后端：

```bash
OTEL_METRICS_EXPORTER=otlp \
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4318 \
VLLM_LOG_STATS_INTERVAL=10 \
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.dapo_math \
  --config rl_dapo_qwen3_4b_math_8k
```

日志记录器使用 OTLP HTTP/protobuf 导出器。如果设置了协议环境变量，`OTEL_EXPORTER_OTLP_METRICS_PROTOCOL` 或 `OTEL_EXPORTER_OTLP_PROTOCOL` 必须为 `http/protobuf`。设置 `OTEL_EXPORTER_OTLP_ENDPOINT` 或 `OTEL_EXPORTER_OTLP_METRICS_ENDPOINT` 之一；指标专用设置优先。

OpenTelemetry SDK 在后台线程导出。导出频率由 `VLLM_LOG_STATS_INTERVAL` 控制，单位为秒（默认：10）。标准 OpenTelemetry 环境变量仍然可用，包括：

- `OTEL_METRIC_EXPORT_TIMEOUT`：每次导出的时间预算，单位为毫秒。
- `OTEL_EXPORTER_OTLP_METRICS_TEMPORALITY_PREFERENCE`：OTLP 聚合时间性（`CUMULATIVE`、`DELTA` 或 `LOWMEMORY`；默认：`CUMULATIVE`）。
- `OTEL_SERVICE_NAME`：附加到导出指标的服务名。
- `OTEL_RESOURCE_ATTRIBUTES`：额外的资源属性。
- `OTEL_SDK_DISABLED=true`：完全禁用 OpenTelemetry。

日志记录器还会附加 `model_name`、`hostname`、`rank`、`local_rank`、`world_size`、`dp_rank`、`tp_rank` 和 `generator_name` 资源属性。