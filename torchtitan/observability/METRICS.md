我们支持自动收集以下指标：
1. 高层系统指标，如 MFU、平均损失、最大损失和每秒词数，以及一些
2. 内存指标，用于衡量最大 VRAM 消耗和 OOM 次数
3. 计时指标，用于衡量数据加载瓶颈

这些指标随后可在 TensorBoard、W&B 或 MLflow 仪表盘中可视化。

## TensorBoard

要通过本地 Web 浏览器可视化远程服务器上训练模型的 TensorBoard 指标：

1. 在训练配方中将 `TensorBoardLogger.Config()` 添加到 `metrics.loggers`。

2. 设置 SSH 隧道，在本地 CLI 中运行以下命令
```
ssh -L 6006:127.0.0.1:6006 [username]@[hostname]
```

3. 在登录远程服务器的 SSH 隧道内，进入 torchtitan 仓库，启动 TensorBoard 后端
```
tensorboard --logdir=./outputs/tb
```

4. 在本地 Web 浏览器中，访问它提供的 URL 或 http://localhost:6006/。

## Weights and Biases

如果使用 `wandb login` 登录，Weights and Biases 会自动将指标发送到远程服务器。

在配方中将 `WandBLogger.Config()` 添加到 `metrics.loggers`。设置
`WANDB_PROJECT` 以选择 W&B 项目；默认为 `torchtitan`。

以 `trainer/step`（训练步数）为横轴绘制指标。生成的图表已默认如此；在保存的面板中，将其选为 x 轴。W&B 自带的 `Step`（`_step`）统计的是日志调用次数。例如：训练和验证都在第 10 步记录日志。这会产生两行，`trainer/step` 均为 10，但 `Step` 值不同。

`WANDB_RESUME_FROM` 和 `WANDB_FORK_FROM` 接受 `<run>?_step=N`，其中 N 是 `Step` 值。要从训练步数 S 开始，请使用 `trainer/step` 为 S 的那一行的 `Step`。

示例可查看 Llama 3 [配方](../../torchtitan_recipes/models/llama3.py)

## MLflow

使用 `pip install mlflow-skinny` 安装可选的跟踪客户端
（或完整的 `mlflow` 包）。

在配方中将 `MLFlowLogger.Config()` 添加到 `metrics.loggers`，并设置 `MLFLOW_TRACKING_URI`
和 `MLFLOW_EXPERIMENT_NAME` 以选择跟踪服务器和实验。

## 配置指标导出器

```python
from torchtitan.observability.metrics import (
    MetricsProcessor,
    TensorBoardLogger,
    WandBLogger,
)

metrics = MetricsProcessor.Config(
    loggers=[WandBLogger.Config(), TensorBoardLogger.Config()],
)
```

日志记录器按列表顺序构建和调用。空列表会禁用指标导出，但保留控制台报告。受监督的处理器将文件写入 `dump_folder/save_tb_folder`；`save_for_all_ranks=True` 会从每个 rank 导出，但 MLflow 始终使用持有损失的 rank。日志记录器构建错误会传播给调用方。

自定义导出器需继承 `BaseLogger`，定义嵌套的
`Config(BaseLogger.Config)`，并实现 `log(metrics, step)` 和 `close()`。
其构造函数接收自身的配置和运行时参数
`log_dir`、`config_dict` 和 `tag`。当导出器必须仅在指标 rank 上运行时，设置 `Config.metrics_rank_only = True`。

RL 指标处理器接受相同的 `loggers` 列表，在控制器上构建，并保留自己的聚合和控制台过滤器。RL 使用相同的 W&B 项目默认值，无需更改环境变量。