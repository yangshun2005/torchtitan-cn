## GraphTrainer

[![集成与数值测试](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer.yaml?query=branch%3Amain) [![GraphTrainer H100 8 GPU 集成测试](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer_h100.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer_h100.yaml?query=branch%3Amain)

本实验通过工具包式使用 PyTorch 编译器技术，演示 torchtitan 中基于图的分布式训练。目标是让用户在大规模分布式训练中，对编译器栈的性能、数值和可调试性拥有明确控制权。动机与设计理念见 [Manifesto](MANIFESTO.md)。

> [!NOTE]
> 欢迎 GraphTrainer 用户和开发者加入 [PyTorch 公共 Slack 频道](https://pytorch.slack.com/archives/C0C4ZRE5BL7)，提问、讨论和获取更新。

**核心特性：**
- **完整训练步图捕获** — 基于 `make_fx` 的 `minimal_fx_tracer` 将前向 + 损失 + 反向（可选 `optimizer.step`）追踪为单个 FX 图，不做 AOTAutograd 分区，从而完整可见并控制整个计算。
- **[SimpleFSDP](https://arxiv.org/abs/2411.00284)** — 基于编译器的 FSDP，将分片表示为计算图内的参数化集合通信，完全可追踪，同时相比 eager FSDP2 提升内存和吞吐量。
- **张量粒度内存策略** — 每个激活可独立选择保存、重算或卸载到 CPU，不同于模块级 eager SAC。不同策略可在同一层内自由混用。
- **图 Pass 流水线** — 分为默认（保持数值）和可选性能 Pass：用于通信/计算重叠的分桶、异步 TP、区域/完整 Inductor 编译、CUDA 图、CPU 卸载和选择性激活重算。
- **预编译（单卡编译）** — 在单 GPU 上编译，序列化构建产物，训练时在所有 rank 加载——完全跳过编译。配置指纹可检测过期产物。
- **可组合并行** — 图中支持 FSDP + TP + EP，异步张量并行（通过对称内存的微流水线 TP）作为可选图 Pass。
- **调试工具** — 集成 tlparse 支持浏览器查看图，profiler trace 中提供 CUDA 图内核标注。

### 前置条件

GraphTrainer 需要最新的 PyTorch nightly，可通过以下命令安装（例如 CUDA 13.0）：
```bash
pip3 install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu130 --force-reinstall
```
可将 `cu130` 替换为其他 CUDA 版本。

### 快速开始

#### 训练 Llama3-8B

```bash
MODULE=torchtitan_recipes.graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh
```
#### 训练 DeepSeek-V3 671B Dist-MoE

```bash
MODULE=torchtitan_recipes.graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_671b_dist_moe_bf16 ./run_train.sh
```

#### 训练 Qwen3-14B

```bash
MODULE=torchtitan_recipes.graph_trainer.qwen3 CONFIG=graph_trainer_qwen3_14b ./run_train.sh
```

### 配置并行

将并行选择放入 recipe。例如：

```python
from torchtitan_recipes.graph_trainer.llama3 import (
    graph_trainer_llama3_8b,
)

def llama3_8b_fsdp4_tp2():
    config = graph_trainer_llama3_8b()
    config.parallelism.data_parallel_shard_degree = 4
    config.parallelism.tensor_parallel_degree = 2
    return config
```

使用 `NGPU=8 MODULE=my_graph_configs
CONFIG=llama3_8b_fsdp4_tp2 ./run_train.sh` 启动。EP 和其他并行轴使用相同模式。

### GraphPP 流水线并行

GraphPP 是 GraphTrainer 模型的流水线并行路径。
它复用 TorchTitan 的 eager PP 模块切分和 PyTorch PP 调度，然后用 GraphTrainer 的 `minimal_fx_tracer` 对每个本地 stage 追踪一个代表性微批次。生成的每 stage 图包复用于后续微批次；`GraphPipelineRuntime` 仅对每个 PP 调度动作执行预构建的可调用对象。

设计参考：
- GraphPP RFC: https://github.com/pytorch/torchtitan/issues/3780
- CUDA 图兼容的 GraphPP RFC: https://github.com/pytorch/torchtitan/issues/3820

```python
config.training.disable_accelerator_graphs = True
config.parallelism.pipeline_parallel_degree = 2
config.parallelism.num_pp_microbatches = 8
config.parallelism.pipeline_parallel_schedule = "Interleaved1F1B"
config.parallelism.data_parallel_shard_degree = 4
config.parallelism.expert_parallel_degree = 2
```

支持的运行时调度包括 `Interleaved1F1B`、`ZBVZeroBubble` 和 `DualPipeV`。GraphPP 构建 stage 本地前向/反向图、可选 FSDP `UNSHARD` / `REDUCE_GRAD` 图、用于拆分反向调度的可选 dI/dW 图，以及用于 `OVERLAP_F_B` 动作的多路复用图。区域和完整 Inductor 编译在提取的 GraphPP 可调用对象上复用现有 GraphTrainer 编译 Pass；GraphPP 特定处理在调用这些 Pass 之前保留在 GraphPP 栈中。多路复用图将前向图保留为目标模块和 ShapeEnv，在其前插入反向占位符/计算，并将反向元数据传入该 ShapeEnv。这样在不修改共享编译 Pass 的情况下，为完整 Inductor 保留前向集合通信大小的来源信息。

GraphPP 遵循与非 PP GraphTrainer tracer 相同的子类边界。提取的图在扁平普通张量叶子上运行。暴露给 PP 运行时的值从 tracer 元数据重新包装：stage 前向输出、发送到前序 stage 的输入梯度，以及赋值给活跃 `param.grad` 之前的参数梯度。内部值保持扁平，因为它们从不离开 GraphPP 图执行：为反向保存的张量、未分片 FSDP 参数、原始梯度叶子、reduce-grad 输入和多路复用中间输出。

当前限制：GraphPP 尚不加载预编译产物，EP 重叠标注将在后续 PR 中与 GraphPP 组合。

### 编译器优化

GraphTrainer 有内置 Pass 流水线，由其编译配置控制。

```python
config.compile.inductor_compilation = "full"
config.compile.numerics_changing_optim = True
config.compile.memory_policy = "full"
config.compile.full_recompute_save_ops = (
    "layers.*.feed_forward.router.gate::aten.mm.dtype | "
    "layers.*.attention.wkv_a::aten.mm.default"
)
config.compile.disable_passes = ["custom_codegen_pass"]
# 禁用所有可选 Pass：
config.compile.enable_passes = False
```

图级 CUDA 图和内核标注 Pass 仍保留在 `cuda_graph.py` 中，但不属于内置 Pass 流水线。自定义调用方可通过 `construct_cuda_graph_passes` 显式应用它们。默认情况下，CUDA 图捕获由客户端围绕完整前向/反向执行负责。用户可通过 `config.training.disable_accelerator_graphs = True` 禁用 CUDA 图。

#### 注册图 Pass

模型或形状特化的重写（例如用手写内核替换子图）是注册图 Pass。在任何模块（包括 torchtitan 之外的模块）中用 `registry.py` 的 `@register_graph_pass` 注册，并在导入它的 recipe 中按名称启用：

```python
# my_pkg/fusions.py
from torchtitan.experiments.graph_trainer.registry import register_graph_pass


@register_graph_pass
def my_kernel_fusion_pass(gm, example_inputs):
    ...  # 用自定义 op 调用替换匹配的子图
    return gm


# recipe
from my_pkg.fusions import my_kernel_fusion_pass

config.compile.extra_passes = [my_kernel_fusion_pass.__name__]
```

选中的 Pass 按列表顺序在联合前向/反向图上运行，位于其他 GraphTrainer Pass 之后、Inductor 编译之前：

```
... -> tag_with_memory_policy -> selective_activation_remat -> bucketing/reordering
  -> [async_tensor_parallel] -> [compile.extra_passes]
  -> annotate_flex_attention -> regional_inductor
```

GraphPP 和 SPMD 梯度累积在将联合图拆分为每动作图之前对其运行这些 Pass。Pass 看到的是前面 Pass 处理后的图：重算的前向 op 已在其反向使用者之前复制，集合通信已分桶并重排。`compile.extra_passes` 中未注册或重复的名称会在 trainer 初始化时抛错。替换节点应复制被替换节点的 meta，其中携带后续 Pass 读取的 fake value、模块 FQN 和前向/反向标记。

### 专家并行重叠

EP 重叠是面向具有真实专家并行集合通信的 MoE 模型的实验性 graph-trainer 优化。仅在 `expert_parallel_degree > 1` 时启用：

```python
config.compile.ep_overlap.enabled = True
config.compile.ep_overlap.chunk_dim = "batch"
config.compile.ep_overlap.module_fqn = "layers.*"
config.parallelism.data_parallel_shard_degree = 4
config.parallelism.expert_parallel_degree = 2
```

支持的 chunk 选择：

- `chunk_dim="batch", module_fqn="layers.*"`
  用于 transformer 块 chunk。
- `chunk_dim="batch", module_fqn="layers.*.feed_forward"`
  用于仅 MoE 的 batch chunk。
- `chunk_dim="seq", module_fqn="layers.*.feed_forward"`
  用于仅 MoE 的序列 chunk。

选中的模块前向在追踪前用 eager chunk 包装；追踪期间记录的 chunk 标注是 EP 重叠调度器消费的契约。

当前限制：

- EP 重叠仅在 `expert_parallel_degree > 1` 时有意义。
- 序列 chunk 仅限 `layers.*.feed_forward`；不支持对 attention 按序列 chunk。
- EP 重叠已针对 graph-trainer 数值和集成套件中测试的 DP/EP 配置验证。与其他重叠或分片模式的可组合性应在使用前通过损失对比验证。

### 实验性 AutoParallel 分片

GraphTrainer 可使用 AutoParallel 为支持的模型求解 SPMD 放置，然后通过常规 GraphRuntime 流程追踪并编译放置后的模型。在 recipe 中设置 `config.compile.enable_autoparallel = True` 启用。

Llama 3 调试模型：

测试的 Llama 3 布局在该 recipe 中还将数据并行分片和张量并行设为 2。

DeepSeek V3 调试模型：

测试的 DeepSeek V3 布局使用数据并行分片 4 和专家并行 2。

AutoParallel 仅负责生成放置后的模型。之后 GraphTrainer 用 `minimal_fx_tracer` 捕获完整训练步，并应用常规图 Pass 流水线：配置的内存策略、选择性激活重算、CPU 卸载、分桶和重叠 Pass、区域或完整 Inductor 编译、CUDA 图兼容性检查，以及其他启用的 GraphTrainer Pass。这使 AutoParallel 放置与手动并行 GraphTrainer 模型使用的相同编译器选项可组合。

AutoParallel 可能选择与手动并行路径不同的分片、集合通信调度和算子 lowering。这会改变数值，因此 AutoParallel 数值测试检查与 eager 基线紧密一致，而非要求逐位相同的损失。

### 预编译（单卡编译）

预编译允许在单 GPU 上编译 AOT 图并保存到磁盘，训练时在所有 rank 加载——完全跳过编译。这使用单卡编译（CooR）生成与 rank 无关的产物。在选中的 recipe 中设置 `compile.precompile_artifact_dir` 即可在两个步骤中启用预编译。

**产物时效性：** 预编译产物与创建时使用的确切 PyTorch 版本、CUDA 版本、模型架构和并行配置绑定。更改任何一项都需要重新生成产物。过期产物通过配置指纹自动检测，并在加载时抛错。升级 PyTorch 或更改模型/并行设置时，请删除旧产物并重新运行预编译。

集成运行器包含完整的 Llama 3 和 DeepSeek V3 预编译 recipe，并运行单进程编译和分布式训练两个阶段：

```bash
python -m torchtitan.experiments.graph_trainer.tests.run_precompile_tests \
    /tmp/precompile_test --ngpu 8
```

<details>
<summary><code>--virtual-local-rank</code> 说明</summary>

此 torchrun 标志使每个 worker 进程看到 `LOCAL_RANK=0` 并指向 `cuda:0`。torchrun 通过 `CUDA_VISIBLE_DEVICES` 隔离每个 worker 的 GPU，因此 `cuda:0` 对每个 worker 映射到不同的物理 GPU。CooR 需要此设置，因为预编译产物是在单进程上针对 `cuda:0` 编译的，而 CooR 在运行时通过 `_runtime_compute_coordinate_on_dim` 动态处理 rank 特定计算。
</details>

预编译适用于任何产生可序列化输出的编译器 Pass，包括 `full_inductor_compilation` 和 `regional_inductor`。多节点设置中，产物目录请使用共享文件系统路径。

### 可组合性支持

GraphRuntime 可组合性状态：

| 特性 | 支持 |
| :--------: | :--------: |
|Meta 初始化| ✅ |
|激活检查点| ✅ |
|激活卸载| 🚧 |
|混合精度训练| ✅ |
|张量并行| ✅ |
|上下文并行| ✅ |
|分布式检查点| ✅ |
|CUDA 图| ✅ |
|MXFP8 训练| 🚧 |
|专家并行| ✅ |
|专家并行 + 激活检查点| 🚧 |
|专家并行 + 流水线并行| 🚧 |
|基于图的流水线并行| 🚧 |
|微批次重叠| 🚧 |
|预编译| 🚧 |

### 支持的模型

GraphTrainer 追踪一个静态形状训练步，因此多模态模型运行纯文本版本，不含视觉编码器。

| 模型 | 说明 |
| :-------- | :-------- |
| Llama 3 | |
| DeepSeek V3 | |
| Qwen3（dense 和 MoE） | |
| Muse Glimmer | 纯文本版本；FSDP、TP 和 PP；无 CP |
| GPT-OSS | FSDP、TP、EP 和 PP；无 CP |

### 引用

如果您觉得 SimpleFSDP 有用，请考虑引用以下论文：

```latex
@article{zhang2024simplefsdp,
  title={SimpleFSDP: Simpler Fully Sharded Data Parallel with torch. compile},
  author={Zhang, Ruisi and Liu, Tianyu and Feng, Will and Gu, Andrew and Purandare, Sanket and Liang, Wanchao and Massa, Francisco},
  journal={arXiv preprint arXiv:2411.00284},
  year={2024}
}
```