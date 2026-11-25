以下性能基准由 PyTorch 团队于 2025 年 6 月完成，用于衡量 async TP 相比原生 TP 基线的性能提升。

### 模型

Llama 3.1 8B、70B

### 硬件

性能基准在 [Grand Teton 平台](https://engineering.fb.com/2022/10/18/open-source/ocp-summit-2022-grand-teton/)上运行，其中
- 每台主机配备 8 块 NVIDIA H100 GPU，通过 NVLink 全互联。
- 每块 H100 GPU 配备 96GB HBM2e，峰值内存带宽 2.4 TB/sec。
- 主机之间通过后端 RDMA 网络互联，每块 GPU 带宽 400 Gb/s。
- 使用默认 500W 功耗上限，但调高至 700W TDP 可能带来进一步加速。


### 结果

详细性能结果和训练配置见下表：

#### Llama3 70b，256 块 H100，FSDP=32，TP=8，torch.compile，完整 AC，本地批大小 16

| 量化      | 原生 TP tokens/sec | Async TP tokens/sec |  Async TP 加速比
| :---------------- | :---- | :---- | :--- |
| 无（bfloat16）   | 597.3 | 652.4 | 1.09 |
| float8 tensorwise | 809.8 | 942.4 | 1.16 |
| float8 rowwise    | 599.6 | 624.8 | 1.04 |

#### Llama3 8b，64 块 H100，FSDP=8，TP=8，torch.compile，逐算子 SAC，本地批大小 12

| 量化      | 原生 TP tokens/sec | Async TP tokens/sec |  Async TP 加速比
| :---------------- | :----- | :----- | :--- |
| 无（bfloat16）   | 4378   | 4809.4 | 1.10 |
| float8 tensorwise | 5078.1 | 5570.1 | 1.10 |
| float8 rowwise    | 3708.5 | 3914.9 | 1.06 |

**注意**：原生 TP float8 rowwise 训练基线性能偏低的问题正在此处处理：https://github.com/pytorch/torchtitan/issues/1207

### 版本与日期

| 仓库 | 提交 | 日期 |
| --- | --- | --- |
| torch | [38410cf9](https://github.com/pytorch/pytorch/commit/38410cf9b57079f3360c1e79601973a01cb2588c) | 2025/06/14 |
| torchao | [6243040](https://github.com/pytorch/ao/commit/6243040807b9ceee889a58cba8e68c5fc4e2ebd8) | 2024/06/13 |
| torchtitan | [820504e](https://github.com/pytorch/torchtitan/commit/820504e20d1149fbf0b98c567af24c4b0433b22d) | 2024/06/13 |