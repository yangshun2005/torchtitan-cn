# 分布式 MoE

TorchTitan 可将标准路由专家替换为 CuTe DSL 的 [`dist-moe`](https://github.com/meta-pytorch/dist_moe) 后端。路由器仍保留在 TorchTitan 中，负责生成 top-k 专家 ID 和分数。Dist-MoE 将 token 分发、分组专家计算、激活存储、暂存存储和合并作为一次有序操作统一管理。

TorchTitan 在 NVIDIA SM100 或更新架构上支持 BF16 和异步 MXFP8 Dist-MoE 训练，专家激活支持 SwiGLU 或带边界的 SiTU-GLU。附录还提供 NVFP4 推理，但 TorchTitan 尚未开放。Dist-MoE 要求 `precision.param_dtype="bfloat16"`，因为 FSDP 以 BF16 反分片其持久参数。

## 安装

Dist-MoE 是可选组件，独立于 TorchTitan 发布。选择 Dist-MoE 配方前，请直接从公开 Git 仓库安装最新 nightly 版本：

```bash
python -m pip install \
  "git+https://github.com/meta-pytorch/dist_moe.git@main"
```

普通 TorchTitan 导入和配方不会导入 Dist-MoE。选中的 Dist-MoE 配方会执行受保护的导入，并在包不可用时报告明确的错误。TorchTitan 的基础依赖有意不安装这个仅限 Blackwell 的后端。

## 选择后端

在一个 transform 上选择专家精度，并在 trainer 上单独配置 rank 级运行时：

```python
from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_16b

config = deepseek_v3_16b()
config.dist_moe_runtime = DistMoeRuntime.Config(
    scratch_capacity_factor=4.0,
    activation_slot_capacity_factor=1.0,
    pp_activation_slot_policy="stage_microbatch",
)
config = apply_transforms(config, [DistMoeTransform()])
```

对于 MXFP8 专家，设置 `expert_precision="mxfp8"`。稠密注意力、共享专家、前馈和语言模型头的线性层仍是独立的量化选择：

```python
import dist_moe

from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_16b

config = deepseek_v3_16b()
config.dist_moe_runtime = DistMoeRuntime.Config(
    scratch_capacity_factor=4.0,
    activation_slot_capacity_factor=1.0,
)
config = apply_transforms(
    config,
    [
        DistMoeTransform(
            expert_precision="mxfp8",
            block_scaled_config=dist_moe.BlockScaledConfig(
                pipeline="staged",
                fast_math=False,
            ),
        )
    ],
)
```

对于 MXFP8 RL 生成器，融合 SwiGLU 的 Mega 内核需要推理权重格式和推理运行时：

```python
import dist_moe

from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
from torchtitan_recipes.rl.alphabet_sort import rl_grpo_qwen3_30b_a3b_varlen

config = rl_grpo_qwen3_30b_a3b_varlen()
config.generator.dist_moe_runtime = DistMoeRuntime.Config(
    inference=True,
    scratch_capacity_factor=4.0,
)
config.generator = apply_transforms(
    config.generator,
    [
        DistMoeTransform(
            expert_precision="mxfp8",
            block_scaled_config=dist_moe.BlockScaledConfig(pipeline="mega"),
            mxfp8_mega_fused_inference=True,
        )
    ],
)
```

仅在 RL 生成器上设置 `mxfp8_mega_fused_inference`，因为它不支持反向传播。验证期间，trainer 改为在 `torch.no_grad()` 下运行。

已验证的 671B 配方位于 `torchtitan_recipes.models.deepseek_v3`。调试和 16B DeepSeek 配方位于 `torchtitan_recipes.tests.models.deepseek_v3`，Kimi K3 调试配方位于 `torchtitan_recipes.tests.models.kimi_k3`。GraphTrainer 遵循对应的 `graph_trainer` 和 `tests.graph_trainer` 归属。配方使用 `_dist_moe_bf16` 和 `_dist_moe_mxfp8` 后缀，并选择兼容 CUDA graph 的 varlen 注意力。DeepSeek MXFP8 配方还会转换稠密投影和语言模型头。

## 配置归属

`DistMoeTransform` 直接将每个标准 `RoutedExperts.Config` 替换为选定的 BF16 或 MXFP8 实现。替换后保留 TorchTitan 结构化的 `GroupedLinear` 参数布局：

- `w13.weight` 形状为 `[E, 2, F, D]`。
- `w2.weight` 形状为 `[E, D, F]`。
- 可选的 `output_postprocess` 仍是普通 TorchTitan 模块。

这保留了参数、优化器、FSDP 和检查点的归属。转换后的模块不构建标准激活或 token 分发器，因为附录在内部执行这些操作。transform 直接从源激活配置推导 SwiGLU 或带边界的 SiTU-GLU，并拒绝其他激活契约。

逐层设置属于 transform：

| 设置 | 含义 |
| --- | --- |
| `expert_precision` | 选择 `"bf16"`（默认）或异步 `"mxfp8"` 专家路径。 |
| `inplace_wgrad_accum` | 让附录 WGRAD 内核直接累加到标准 `parameter.grad` 存储。BF16 和 MXFP8 专家默认启用。GraphTrainer 配方暂时禁用它，因为 tracing 会将参数函数化为图输入；图 pass 必须先将这些输入绑定到其叶子 `parameter.grad` 目标，才能启用累加。 |
| `bf16_grouped_gemm_preset` | 可选的 BF16 FPROP/DGRAD 调度覆盖，供专家用户使用。`None` 使用附录的形状感知生产默认值。BF16 WGRAD 有自己的生产调度。 |
| `block_scaled_config` | 仅限 MXFP8 的附录策略。`pipeline="staged"` 使用独立专家内核；`pipeline="mega"` 使用融合的分块流水线实现。`fast_math` 选择近似融合 SwiGLU sigmoid 数学，`kernel_config` 是仅限专家的 CuTe 调优覆盖。 |
| `mxfp8_mega_fused_inference` | 仅限 MXFP8 RL 生成器；要求 `pipeline="mega"`。以交错格式准备权重，使 Mega 内核能将 SwiGLU 融合进 W13 GEMM，仅用于推理。量化权重和缩放与训练数值完全一致。 |

一个 `DistMoeRuntime.Config` 拥有每个本地 Dist-MoE 层共享的 rank 级资源：

| 设置 | 含义 |
| --- | --- |
| `activation_slot_bytes` | 每个活跃激活槽请求的精确保存前向状态容量，不含暂存。使用此专家级字节覆盖时，将 `activation_slot_capacity_factor=None`。 |
| `activation_slot_capacity_factor` | 每槽保存状态容量相对于均衡路由的倍数。`1.0` 在聚合槽使用均衡时保留每个符合条件的中间结果。默认 `None`，与 `activation_slot_bytes` 互斥。 |
| `scratch_capacity_factor` | 必须完全放入 HBM 暂存的路由不均衡量。`1.0` 覆盖均衡的 `local_tokens * top_k` 路由。 |
| `vmm_capacity_factor` | 可选的设备加主机暂存总量上限。`None` 禁用 VMM；大于 `scratch_capacity_factor` 的值提供主机支持的溢出容量。保存的激活仍留在 HBM。 |
| `pp_activation_slot_policy` | `"stage_microbatch"` 在 stage-microbatch 生命周期复用槽；`"microbatch"` 为一个 microbatch 在所有本地 stage 间保留一个更深的槽。默认 `"stage_microbatch"`。 |

配方将此配置赋给可选的 `TrainingEngine.Config.dist_moe_runtime` 字段，其基础类型为 `Configurable.Config | None`。具体配置的 `build()` 方法仅在配方选择它时构建 `DistMoeRuntime`。前向/反向初始化向运行时提供物化的模型部分、拓扑、调度和执行模式注册。模型维度、精确本地 token 数、专家元数据、活跃槽、每槽层数和内核精度均为推导得出，而非在配方中重复。Eager 执行不设置附录 WGrad dtype，因此附录从每个活跃参数声明的梯度 dtype、现有梯度存储或参数 dtype 解析。GraphTrainer 对其函数式 WGrad 输出使用 BF16 混合精度参数 dtype，因为准备好的 MXFP8 图操作数不暴露活跃参数目标。当前 FSDP 将生成的本地 BF16 WGrad 转换为 `precision.reduce_dtype` 进行归约。如果 FSDP 日后声明 FP32 参数梯度 dtype，eager Dist-MoE 将直接使用，无需单独的 TorchTitan 覆盖。

## 暂存容量

设 `T` 为 CP 和序列并行分片后的精确本地输入 token 数，`K` 为 top-k，`P` 为专家并行度。均衡路由在每个 rank 上接收 `T * K` 行。最大可能接收数为：

```text
T * P * min(K, num_local_experts)
```

相对于均衡路由的最坏情况不均衡为：

```text
P * min(K, num_local_experts) / K
```

`scratch_capacity_factor` 设置 HBM 驻留的接收和暂存上限。它不会截断或重新平衡路由。因子 `1.0` 适用于强制均衡路由。真实路由需要实测余量；因子四是有意的配方选择，不是通用默认值。块缩放内核还会将每个本地专家填充到其 M-tile 大小，因此计划行数可能略大于逻辑上限。

不使用 VMM 时，超过设备因子即报错。使用 VMM 时，`vmm_capacity_factor` 是更大的设备加主机正确性上限。超过该上限仍会报错。

## 保存的激活与流水线槽

附录分配四个不同区域：

| 分配 | 位置 | 内容与生命周期 |
| --- | --- | --- |
| 对称通信缓冲区 | 可对等寻址的 HBM | 路由元数据、信号、分发输入、合并输出和梯度，生命周期为上下文。 |
| 保存激活区域 | Rank 本地 HBM | 强制层输入和选择保存的专家中间结果，划分为活跃激活槽。 |
| 设备暂存 | Rank 本地 HBM | 前向临时、重算临时和激活梯度，一次由一个本地层操作复用。 |
| VMM 溢出暂存 | 同一 CUDA 虚拟范围内的固定主机内存 | 可选的暂存需求，超过设备因子且在总因子内。 |

无 PP 时，一个激活槽包含所有本地 MoE 层。使用 eager PP 时，PyTorch 在预热前分析最终调度，将每个活跃 `(stage, microbatch)` 区间分配给可复用槽。默认 `"stage_microbatch"` 策略按最大本地 stage 设置每个槽的大小，并在该 stage 反向释放保存状态后复用。`"microbatch"` 策略为一个 microbatch 在所有本地 stage 间保留一个槽，因此按它们合并的 MoE 深度设置每个槽的大小。

激活容量为每个活跃槽独立配置。设置 `activation_slot_bytes` 获得精确逻辑字节预算，或使用 `activation_slot_capacity_factor` 在强制层输入之上缩放可选保存状态。两个控件互斥。当两者均为 `None` 时，附录选择其最小正确计划：每个分配的层输入保持保存，其他专家中间结果在反向中动态重算。提高任一策略可保留更多中间结果并减少重算，直至规划器记录的最大有用每槽预算。

PyTorch 以 stage 和 microbatch 索引调用每个 eager 流水线 stage 注册的前向上下文。`DistMoeRuntime` 使用预计算分配在模型执行前选择附录激活槽。每个槽按 `max_moe_layers_per_activation_slot` 规划，因此只有槽 ID 变化。这不会添加模型 kwargs，也不需要自定义流水线 stage 子类。

GraphPP 复用同一运行时拥有的槽解析器，而不进入 eager stage 上下文。追踪期间，TorchTitan 提供一个显式 `activation_slot_id_1` 输入，并重写精确的 Dist-MoE 前向操作以使用它。执行时，承载前向的调度动作解析其预计算物理槽，并传递对应的不可变设备标量视图。这为每个 stage 生成一个图，无需模型 kwargs、每 microbatch 图变体或设备端索引操作。Dist-MoE 反向使用其保存的前向状态，因此不需要第二个槽输入。

附录的函数式和累加式反向操作对非严格 FX 追踪均可见。原地累加将其目标显式传递给注册的反向操作，因此 eager 和图执行保留相同的标准 `parameter.grad` 归属契约。

## VMM 溢出暂存

VMM 将快速暂存前缀保留在 HBM 中，并将额外的固定主机页映射到同一稳定的 CUDA 虚拟范围：

```python
from torchtitan.config.transform import apply_transforms
from torchtitan.config.transform.dist_moe import DistMoeTransform
from torchtitan.models.common.dist_moe.runtime import DistMoeRuntime
from torchtitan_recipes.tests.models.deepseek_v3 import deepseek_v3_16b

config = deepseek_v3_16b()
config.dist_moe_runtime = DistMoeRuntime.Config(
    activation_slot_capacity_factor=1.0,
    scratch_capacity_factor=1.0,
    vmm_capacity_factor=4.0,
)
config = apply_transforms(config, [DistMoeTransform()])
```

超过因子一且至多因子四的暂存可使用主机支持的页。保存的激活不会移至主机内存。运行时使用附录的默认预取生命周期，在上下文创建期间将物理 VMM 分配与通信缓冲区初始化重叠，而不改变容量、稳态放置、内核行为或执行时同步。

## FSDP、MXFP8 与重算

BF16 权重使用普通 FSDP 生命周期。MXFP8 专家复用 TorchTitan 的准备权重张量生命周期：

1. FSDP 以 BF16 全收集持久高精度分片。
2. 全收集后钩子准备分组 32x32 MXFP8 qdata 和 FPROP/DGRAD 缩放布局。
3. FSDP 释放临时 BF16 通信存储。
4. Dist-MoE 在该反分片生命周期内使用准备好的操作数。
5. FSDP 在正常重分片边界释放准备好的存储。

使用 `reshard_after_forward=False` 时，流水线 microbatch 在反向中复用一个准备好的权重。使用 `reshard_after_forward=True` 时，前向和反向执行单独的反分片和准备。无 FSDP 时，模块为每次调用直接准备活跃参数。

完整的 `dist_moe.routed_experts` 调用是一个有序操作，包裹在非重算的 remat 区域中。Dist-MoE 拥有其设备端保存或重算专家中间结果的决定；外部激活检查点策略不得重复分发、通信或 arena 变更。

## 专家输出后处理

通用路由专家 API 可拥有一个 `output_postprocess` 模块，在 W2 之后、分数加权 top-k 合并之前运行。TorchTitan 保留该模块的参数、优化器、检查点和 FSDP 归属。每次前向时，Dist-MoE 适配器将其当前参数转换为附录的类型化 `RMSNormPostprocess` 执行描述符。不支持的后处理器在配置期间失败，而不是在合并后运行或回退到非融合回调。

## 运行时生命周期

TorchTitan 通过前向/反向设置拥有一个可选的 Dist-MoE 运行时：

1. 配方将 `DistMoeRuntime.Config` 赋给 `TrainingEngine.Config.dist_moe_runtime`。
2. 模型 transform 独立替换路由专家模块。
3. 前向/反向初始化开始时，标准或图引擎提供最终模型部分、拓扑、活跃性调度及其窄前向上下文注册能力。
4. 运行时创建一个附录上下文，推导槽分配，安装 eager stage 钩子或 GraphPP 解析器，保留移除归属，并将非拥有引用附加到每个本地 Dist-MoE 模块。
5. GraphPP 将解析器的不可变槽视图作为显式图输入提供；eager PP 进入注册的 stage 上下文。
6. 部分初始化回退运行时拥有的资源。正常拆除移除注册、分离模块并关闭附录上下文。

RL vLLM 生成器拥有一个推理运行时（`DistMoeRuntime.Config(inference=True)`，作为 `Generator.Config.dist_moe_runtime` 传入）：仅暂存规划，按 vLLM 的每步 token 预算设置大小。vLLM 在末尾填充每一步（TP 舍入、CUDA graph 大小、DP 最大值）；runner 发布未填充计数，模型将 `padding_mask` 向下传递给路由专家。

## 填充行

在训练和推理中，`padding_mask` 标记的行获得专家 ID `-1` 和零分数。Dist-MoE 从不分发 `-1` 路由，因此填充不消耗通信或专家 GEMM 工作，并产生零输出行。每个 EP rank 必须向调用传递相同的行数：

- 训练原样传递其 microbatch 行。
- 推理调用使用 vLLM