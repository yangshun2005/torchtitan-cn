# 选择专家并行后端

TorchTitan 支持两种执行路由专家的方式：先经 token 分发器再分组专家计算，或使用 Dist-MoE（统一负责分发、专家计算、激活存储与合并）。路由器和检查点参数名仍保留在 TorchTitan 中。

## 选择

| 工作负载 | 后端 | 约束 |
| --- | --- | --- |
| 正确性参考或新加速器 | `AllToAllTokenDispatcher` | 使用 PyTorch 集合通信。可变 token 数需要 CPU 同步；需禁用加速器图。 |
| 搭配参考分发器的量化分组 GEMM | `TorchAOTokenDispatcher` | 为 all-to-all 增加按专家填充。也支持 EP=1。 |
| 使用独立分组 GEMM 实现的 NVIDIA 训练或推理 | `DeepEPTokenDispatcher` | 需要 DeepEP v2。训练使用紧凑分发并带 CPU 同步；需禁用训练加速器图。`no_grad` 下的推理可用 `cuda_graph_compatible=True` 采用静态扩展布局。 |
| 带图捕获 EP 通信的 NVIDIA 训练，包括图训练器 | `HybridEPTokenDispatcher` | 需要 HybridEP 分支及受支持的 NVLink/网络拓扑。设置 `non_blocking_capacity_factor` 实现无 CPU 分发。 |
| 集成通信与计算的 Blackwell BF16 或 MXFP8 训练 | Dist-MoE | 需要 SM100 或更新硬件，且 `precision.param_dtype="bfloat16"`。请先阅读 [Dist-MoE 配方与运行时指南](../dist_moe/README.md)。 |

无专家并行时，`LocalTokenDispatcher` 仅重排本地 token。DeepEP 和 HybridEP 要求 EP>1。TorchTitan 不提供独立的 DeepEP v1 或直连 NCCL EP 分发器。

HybridEP 由 NVIDIA 开发，支持 CUDA 图，配有 TorchTitan [图训练器](../../../experiments/graph_trainer/README.md)配方和集成测试。DeepEP 的图训练器兼容性尚未在此验证；其当前训练适配器的 CPU 同步会阻止 CUDA 图捕获。

DeepEP v2 和 HybridEP 均使用 NVIDIA 专用内核，需要受支持的通信网络。DeepEP v2 需要 Hopper（SM90）或兼容的更新 GPU；HybridEP 使用 TMA，面向 Hopper/Blackwell 系统。其他加速器请先使用 PyTorch all-to-all 参考实现，并在选择优化后端前确认上游硬件与网络要求。

按硬件、精度、图需求和内存预算选择，然后在实际路由分布上测量训练步时间和峰值内存。仅靠分发/合并微基准无法衡量与专家 GEMM 的重叠或激活重计算成本。至少运行 10 个训练步，避免启动开销主导对比。

DeepEP 分数放置、偏置专家、后处理及检查点依赖的图示见[路由器分数吸收](ROUTER_SCORE_ABSORPTION.md)。

## 配置分发器

使用 `TokenDispatcherTransform` 替换现有配方中的分发器：

```python
from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.token_dispatcher import TokenDispatcherTransform
from torchtitan.models.common.moe.token_dispatcher import HybridEPTokenDispatcher
from torchtitan_recipes.models.deepseek_v3 import deepseek_v3_671b

config = deepseek_v3_671b(seq_len=4096)
config = apply_transforms(
    config,
    [
        TokenDispatcherTransform(
            dispatcher=HybridEPTokenDispatcher,
            num_max_tokens_per_rank=(
                config.training.num_tokens_per_microbatch_per_dp_rank
                // (
                    config.parallelism.context_parallel_degree
                    * config.parallelism.tensor_parallel_degree
                )
            ),
            kwargs={"non_blocking_capacity_factor": 1.0},
        ),
    ],
)
```

`num_max_tokens_per_rank` 限定单个 EP rank 在其通信缓冲区生命周期内每次前向分发的输入 token 数。它不是 top-k 扇出后的 token 数。推理引擎必须使用自己的每 rank token 预算。Dist-MoE 使用 `DistMoeTransform` 及其运行时，而非此 transform。

对于图捕获训练，还需提供固定形状的注意力元数据并按现有 HybridEP 配方启用加速器图。仅有无 CPU 分发器不足以让整个模型可捕获。

### HybridEP 容量

- `None`：阻塞式分发按实际 token 数确定置换输出大小。
- `1.0`：非阻塞式分发预留最坏情况下的无填充 token 扇出，内存开销更高。
- 低于 `1.0`：较小的置换输出可能在溢出时丢弃 token。负载均衡不保证任意路由都处于容量内。

容量因子决定置换输出大小，而非持久通信缓冲区。量化专家还需合适的 `pad_multiple`。按专家填充即使在因子 `1.0` 时也可能超出分配。容量低于 `1.0` 或使用填充非阻塞分发时，请在步后检查 `check_hybridep_over_budget()`，并在两次测量间重置标志；该标志会一直保留在设备上直到被显式读取。

## 激活重计算

默认重计算策略保存分发器通信并重算路由专家的 `w13` 投影。要在自定义策略下保留通信：

```python
from torchtitan.distributed.activation_remat import ActivationRematConfig

config.activation_remat = ActivationRematConfig(
    save_regions=["*token_dispatcher.ep_communication"],
)
```

DeepEP 使用 `ep_communication` 控制分发重放，并始终保存完整的合并区域（含反置换）。完整重计算以确定性接收顺序重放分发，增加一次排序。HybridEP 在 `ep_communication` 下同时保存或重放分发与合并；仅选择其 `.dispatch` 或 `.combine` 后缀无法控制这一对。它通过前缀扫描推导 token 位置。all-to-all 分发器则拥有独立的 `dispatch` 和 `combine` 区域，包含本地重排和集合通信。

Dist-MoE 始终保留其外层区域，并自行决定内部激活的保存/重算。策略细节见 [ACTIVATION_REMAT.md](../../../distributed/ACTIVATION_REMAT.md)。

## 依赖

DeepEP v2 和 HybridEP 目前位于同一 `deep_ep` 包的不兼容分支上。对比时请使用独立环境。请查阅上游安装与网络要求：

- [DeepEP v2](https://github.com/deepseek-ai/DeepEP)
- [HybridEP](https://github.com/deepseek-ai/DeepEP/blob/hybrid-ep/docs/README_Hybrid-EP.md)
- [Dist-MoE](../dist_moe/README.md)