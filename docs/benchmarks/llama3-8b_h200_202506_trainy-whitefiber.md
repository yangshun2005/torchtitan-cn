本次基准测试由 Trainy 团队于 2025 年 6 月在 WhiteFiber 上执行，用于获取 Trainy 平台在 H200 多主机环境下的性能基线。

### 模型

Llama 3.1 8B

### 硬件

每台主机配置：

- 8 块 NVIDIA H200 GPU，通过 NVLink 互联。
- 主机之间通过后端 RDMA 网络互联，每块 GPU 带宽 400Gb/s（Mellanox CX-7）。

### 配置

运行时 `NUM_NODES` 分别设为 `4` 和 `8`，命令如下。

**警告**：以下命令反映的是本次基准测试时的原始调用方式。torchtitan CLI 此后已改为使用 `--module` 和 `--config` 标志，不再使用 `--job.config-file`。最新用法请参阅当前 [README](/README.md)。
```
  torchrun \
    --nnodes $NUM_NODES  \
    --nproc_per_node 8 \
    --rdzv_id 101 \
    --rdzv_backend c10d \
    --rdzv_endpoint "$MASTER_ADDR:29500" \
    torchtitan/train.py \
    --job.config-file torchtitan/models/llama3/train_configs/llama3_8b.toml \
    --metrics.enable_wandb \
    --training.local_batch_size=2 \
    --training.compile \
    --model.converters="quantize.linear.float8" \
    --quantize.linear.float8.enable_fsdp_float8_all_gather \
    --quantize.linear.float8.precompute_float8_dynamic_scale_for_fsdp \
    --quantize.linear.float8.force_recompute_fp8_weight_in_bwd \
    --profiler.profile_freq 1000000
    --training.steps 2000
```

### 结果

详细的性能结果和训练配置见下表，也可在[此 WandB 报告](https://api.wandb.ai/links/asaiacai/w4c46stp)中查看。`TPS` 和 `Memory(GiB)` 均在第 100 次迭代时任意采样：

| NUM_NODES | TPS/GPU | Memory(GiB) |
| ----- | ----: | ----: |
| 4 | 10938 | 47.96 |
| 8 | 10753 | 46.97 |


### 版本与日期

| 仓库 | 提交 | 日期 |
| --- | --- | --- |
| torch | [2.8.0a0+5228986c39](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-25-05.html) | 2025/05/29 |
| torchao | [0afa4c1](https://github.com/pytorch/ao/commit/0afa4c1bd28c82921e360ddbd1b27c9d6da5b947) | 2025/06/13 |
| torchtitan | [e7c0cae](https://github.com/pytorch/torchtitan/commit/e7c0cae934df78d6e9c2835f42ff1f757dc3fddc) | 2025/06/13 |