以下性能基准由 `torchtitan` 团队于 2024 年底使用最新版本的 `torch`、`torchao` 和 `torchtitan` 完成。

### 模型

Llama 3.1 8B、70B、405B

我们通过在 Llama 3.1 8B、70B 和 405B 模型上的实验，展示了 torchtitan 弹性分布式训练的有效性：从 1D 并行到 4D 并行，规模从 8 块 GPU 扩展到 512 块 GPU。

### 硬件

性能基准在 [Grand Teton 平台](https://engineering.fb.com/2022/10/18/open-source/ocp-summit-2022-grand-teton/)上运行，其中：
- 每台主机配备 8 块 NVIDIA H100 GPU，通过 NVLink 全互联。
- 每块 H100 GPU 配备 96GB HBM2e，峰值内存带宽 2.4 TB/sec。
- 主机之间通过后端 RDMA 网络互联，每块 GPU 带宽 400 Gb/s。
- 使用默认 500W 功耗上限；调高至 700W TDP 可能带来进一步加速。

### 结果

需要说明的是，在整个实验过程中，内存读数在整个训练过程中保持稳定[^1]；而吞吐量数据（TPS/GPU）每 10 次迭代计算并记录一次，且始终读取第 90 次迭代（任意选定）的值。

我们不报告模型 FLOPS 利用率（MFU），因为启用 Float8 时（作用于 `nn.Linear` 模块），BFLOAT16 Tensor Core 和 FP8 Tensor Core 都参与模型训练，但两者峰值 FLOPS 不同，此场景下 MFU 的定义并不明确。需要指出，1D Llama 3.1 8B 模型在 8 或 128 块 H100 GPU 上不使用 Float8 训练时，MFU 达到 33% 至 39%[^2]（分别为使用和不使用 torch.compile）。

**表 1** 1D 并行（FSDP）。Llama 3.1 8B 模型。8 块 GPU。本地批大小 2，全局批大小 16。选择性激活检查点。

| 技术 | TPS/GPU | 内存(GiB) |
| ----- | ----: | ----: |
| FSDP | 5,762 | 82.4 |
| FSDP + torch.compile | 6,667 | 77.0 |
| FSDP + torch.compile + Float8 | 8,532 | 76.8 |

**表 2** FSDP + CP + torch.compile + Float8。Llama 3.1 8B 模型。8 块 GPU。本地批大小 1。完全激活检查点。

| 并行方式 | 序列长度 | TPS/GPU | 内存(GiB) |
| ----- | ----: | ----: | ----: |
| FSDP 8, CP 1 | 32768 | 3,890 | 83.9 |
| FSDP 4, CP 2 | 65536 | 2,540 | 84.2 |
| FSDP 2, CP 4 | 131072 | 1,071 | 84.0 |
| FSDP 1, CP 8 | 262144 | 548 | 84.5 |

**表 3** 1D 并行（FSDP）。Llama 3.1 8B 模型。128 块 GPU。本地批大小 2，全局批大小 256。选择性激活检查点。

| 技术 | TPS/GPU | 内存(GiB) |
| ----- | ----: | ----: |
| FSDP | 5,605 | 67.0 |
| FSDP + torch.compile | 6,514 | 62.0 |
| FSDP + torch.compile + Float8 | 8,380 | 61.8 |

**表 4** 2D 并行（FSDP + TP）+ torch.compile + Float8。Llama 3.1 70B 模型。256 块 GPU（FSDP 32，TP 8）。本地批大小 16，全局批大小 512。完全激活检查点。

| 技术 | TPS/GPU | 内存(GiB) |
| ----- | ----: | ----: |
| 2D | 829 | 71.9 |
| 2D + AsyncTP | 876 | 67.6 |

**表 5** 3D 并行（FSDP + TP + PP）+ torch.compile + Float8 + AsyncTP。Llama 3.1 405B 模型。512 块 GPU（FSDP 8，TP 8，PP8）。本地批大小 32，全局批大小 256。完全激活检查点。

| 调度方式 | TPS/GPU | 内存(GiB) |
| ----- | ----: | ----: |
| 1F1B | 100 | 82.5 |
| Interleaved 1F1B | 128 | 72.7 |

**表 6** 4D 并行（FSDP + TP + PP + CP）+ torch.compile + Float8 + AsyncTP + 1F1B。Llama 3.1 405B 模型。512 块 GPU（TP 8，PP8）。本地批大小 8。完全激活检查点。

| 并行方式 | 序列长度 | TPS/GPU | 内存(GiB) |
| ----- | ----: | ----: | ----: |
| FSDP 8, CP 1 | 32768 | 76 | 75.3 |
| FSDP 4, CP 2 | 65536 | 47 | 75.9 |
| FSDP 2, CP 4 | 131072 | 31 | 77.1 |
| FSDP 1, CP 8 | 262144 | 16 | 84.9 |

### 版本与日期

| 仓库 | 提交 | 日期 |
| --- | --- | --- |
| torch | [1963fc8](https://github.com/pytorch/pytorch/commit/1963fc83a1c32e162162e2414f78b043f0674bae) | 2024/12/23 |
| torchao | [eab345c](https://github.com/pytorch/ao/commit/eab345c2268a7506355d506ebfc27b5d28e5e7d0) | 2024/12/23 |
| torchtitan | [9dec370](https://github.com/pytorch/torchtitan/commit/9dec370ad26b5f8e9a7333a0e36165018262644b) | 2024/12/26 |


[^1]: 不同 PP rank 的峰值内存使用可能不同。我们取所有 GPU 中的最大值。

[^2]: 测试中使用的是基于 HBM2e 的 SXM H100，TDP 较低，实际峰值 TFLOPs 介于 SXM 和 NVL 之间，具体数值未知。因此该 MFU 数值低于实际 MFU，因为我们直接使用了 SXM 的峰值数据。