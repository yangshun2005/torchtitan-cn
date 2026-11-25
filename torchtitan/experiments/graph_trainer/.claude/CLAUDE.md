# GraphTrainer 开发指南

本文补充根目录 `.claude/CLAUDE.md` 中的核心 torchtitan 约定（代码风格、命名、测试、PR 要求等）。除非下文另有覆盖，否则那里的规则同样适用。

## 图 Pass 签名

所有图 Pass 必须遵循此签名：
```python
def my_pass(gm: torch.fx.GraphModule, example_inputs, *, other_kwargs) -> torch.fx.GraphModule:
```
- 前两个位置参数始终为 `(gm, example_inputs)`。
- 任何额外参数必须是**仅关键字参数**（位于 `*` 之后）。
- Pass 必须返回（可能已转换的）`GraphModule`。
- 不需要 `example_inputs` 的 Pass 仍应接受它（使用 `example_inputs=None`）。

## Pass 配置

每个 Pass 的配置（例如 cudagraph 的 `static_input_indices`）必须在 `construct_default_graph_passes` 中通过 `functools.partial` 在构造 Pass 时绑定，**不要**作为参数经由 `apply_graph_passes` 传递。apply 函数是通用的 Pass 运行器，不得包含 Pass 专属参数。

## Pass 层级

图 Pass 分为三层：

1. **默认 Pass**（`passes.py`、`remove_noop_passes.py` 等）——始终应用。这些 Pass 保持数值不变：清理、内存策略、分桶、异步 TP、FlexInnerAttention 区域 Inductor（与 eager 逐位匹配所必需）。

2. **性能 Pass**（`performance_passes.py`）——通过 `config.compile.numerics_changing_optim = True` 选择性启用。这些 Pass 提升性能，但相比未编译路径可能改变数值（例如 RMSNorm Inductor 融合）。

3. **注册 Pass**——通过 `config.compile.extra_passes` 按名称选择性启用。可从任意模块（包括 torchtitan 之外）用 `@register_graph_pass`（`registry.py`）注册。它们在联合图上运行，位于其他 GraphTrainer Pass 之后、Inductor 编译之前。用于模型或形状特化的重写，例如用手写 kernel 替换子图。

新增 Pass 时，若改变数值则放入 `performance_passes.py`；否则放入 `passes.py` 或 `remove_noop_passes.py` 之类的专用文件。仅部分 recipe 需要的模型或 kernel 特化重写应作为注册 Pass。

## 内存策略框架

PyTorch 的模块级 `torch.utils.checkpoint` 和 eager SAC 对整个模块的输出做粗粒度的保存或重算决策，且激活检查点与 CPU offload 组合困难。本框架改为在 FX 图上以单个张量为粒度操作：每个激活可独立选择保存、重算或 offload，不同策略可在单层内自由混用。

`tag_with_memory_policy_pass` 是统一入口，分两步：

1. **标记节点。** 每个被保存的前向激活标记为以下之一：
   - `MUST_SAVE`——将激活保留在 GPU 内存中。
   - `MUST_RECOMPUTE`——丢弃并在反向时重算。
   - `MUST_CPU_OFFLOAD`——offload 到 CPU，反向之前重新加载。

   标记可手动完成（逐节点注解），也可用高级求解器算法，根据内存预算和计算成本优化保存/重算/offload 的划分。

2. **按标记执行。** 标记后无条件运行两个 Pass（当没有节点携带相关标记时，两者均为空操作）：
   - `apply_cpu_offload_pass`——为 `MUST_CPU_OFFLOAD` 节点插入 offload/reload/wait 操作。
   - `selective_activation_remat_pass`——在其反向消费者之前复制 `MUST_RECOMPUTE` 操作，并删除消费者全在反向的原始操作。

`config.compile.memory_policy` 字段选择标记策略。新策略（例如预算感知的混合 SAC + offload）应作为 `tag_with_memory_policy_pass` 中的新分支添加。

**CPU offload 的 NUMA 绑定：** 在多 NUMA 机器上（例如 GB200 NVLink-C2C），D2H/H2D 带宽在 NUMA 本地约 350 GB/s，跨 NUMA 约 120 GB/s。`Trainer` 在初始化时于 CUDA 硬件上自动应用 NUMA 绑定（`AffinityMode.NODE`），将每个 worker 固定到其 GPU 所在的 NUMA 节点。在非 CUDA 硬件或 `numactl` 不可用时优雅回退。

**检查标记：** `log_activation_memory_policy`（`log_activation_memory_policy.py`）打印所有被反向消费的前向节点，按层分组并合并相同模式。显示内存、dtype、策略（SAVE/RECOMPUTE/OFFLOAD）、形状、子模块、目标操作和源位置。它在 `tag_with_memory_policy_pass` 末尾自动运行，同时记录到 `logger.debug` 和 tlparse（通过 `trace_structured`）。

## 不要为本实验修改核心代码

不要在 `torchtitan/train.py` 或其他核心文件中添加 `if graph_trainer:` 分支。GraphTrainer 通过子类化扩展 `Trainer` 并覆盖行为。


### 本地开发（调试模型，8 GPU）

**从仓库根目录运行所有命令。** 使用根目录的 `./run_train.sh`，设置 `MODULE=graph_trainer.llama3`（或 `.deepseek_v3`、`.qwen3`）。8B/16B 配置使用相对于仓库根目录的 `hf_assets_path`。

需要 `--virtual-local-rank` 的 CooR 预编译工作流，请改用 `torchtitan/experiments/graph_trainer/run_train_precompile.sh`。

```bash
# Llama3 using the topology defined by the recipe
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_debugmodel \
    ./run_train.sh

# DeepSeek-v3 using the topology defined by the recipe (requires H100)
NGPU=8 MODULE=graph_trainer.deepseek_v3 CONFIG=graph_trainer_deepseek_v3_debugmodel \
    ./run_train.sh
```

如需不同拓扑，请添加派生的 Python recipe 并更新其 `parallelism` 字段，而不要在命令行传递配置标志。

### 测试

```bash
# Unit tests (GPU)
pytest torchtitan/experiments/graph_trainer/tests/test_passes.py -x
pytest torchtitan/experiments/graph_trainer/tests/test_precompile.py -x
pytest torchtitan/experiments/graph_trainer/tests/test_trace_module.py -x
pytest torchtitan/experiments/graph_trainer/tests/test_numerics.py -x
pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -x

# Integration tests (8 GPUs)
python torchtitan/experiments/graph_trainer/tests/integration_tests.py <output_dir> \
    --test_suite graph_trainer_default --ngpu 8
```

### 调试图 Pass

每个图 Pass 始终被插桩：记录每个 Pass 的耗时和操作数差异摘要，并将前后图转储到 tlparse。使用 `TORCH_TRACE` 运行，并用 `tlparse` 在浏览器中检查图。

```bash
TORCH_TRACE=/tmp/trace NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b ./run_train.sh
```

### 转储 GraphModule 以调试

要在任意时刻检查 `GraphModule`，将其转储到临时文件：

```python
from pathlib import Path
import tempfile

def dump_gm(gm: torch.fx.GraphModule, name: str) -> None:
    output_path = Path(tempfile.gettempdir()) / f"{name}.txt"
    output_path.write_text(
        gm.print_readable(
            print_output=False,
            include_stride=True,
            include_device=True,
            expanded_def=True,
        )
    )
    print(f"Dumped graph to {output_path}")
```

调试图 Pass 时，在 Pass 前后转储图并对比两个文件，查看具体变化：

```python
def my_pass(gm, example_inputs):
    dump_gm(gm, "my_pass_before")
    # ... transform gm ...
    dump_gm(gm, "my_pass_after")
    return gm
```

```bash
diff /tmp/my_pass_before.txt /tmp/my_pass_after.txt
```

### 在编译函数内打印和检查张量

要在 `torch.compile` *内部*检查张量值或梯度而不产生图中断：

- **简单打印**：`torch._higher_order_ops.print("norm={}", x.norm())`——格式化字符串，仅前向。DTensor 会打印每个 rank 的本地视图，并自动添加 `[rank N]` 前缀。
- **梯度范数**：`from torch.utils.debug_log import debug_grad_log`——对中间量（非直接图输入）调用；在反向时触发并记录每个张量的梯度范数。
- **自定义逻辑**（任意 Python、文件日志、rank 过滤、前向+反向）：将 `@leaf_function` 与 `torch._dynamo.decorators` 中的 `@fn.register_multi_grad_hook` 组合，参照 `torch/utils/debug_log.py` 中的模式：

  ```python
  from torch._dynamo.decorators import leaf_function

  @leaf_function
  def log_tensor(x, tag=""):
      return None  # no-op in forward

  @log_tensor.register_fake
  def log_tensor_fake(x, tag=""):
      return None

  @log_tensor.register_multi_grad_hook
  def log_tensor_hook(x_grad, tag=""):  # non-tensor args passed through unchanged
      print(f"[{tag}][bwd] grad_norm={x_grad.norm():.4f}")

  log_tensor(x, "after_add")
  ```

### 基准测试

使用 `./run_train.sh` 配合基准 recipe，设置少量步数并禁用 tensorboard、profiling 和 flight recorder。已检入的 Llama3 autoresearch recipe 已设置 FSDP=4、TP=2、20 步及这些日志控制：

```bash
# Llama3 8B GraphRuntime (8xH100, FSDP+TP, 20 steps)
NGPU=8 MODULE=graph_trainer.llama3 CONFIG=graph_trainer_llama3_8b_autoresearch \
    ./run_train.sh

# A DeepSeek-v3 benchmark should use a derived recipe with the corresponding
# parallelism and logging fields set in Python.
NGPU=8 MODULE=my_graph_configs CONFIG=deepseek_v3_16b_benchmark ./run_train.sh
```

查看**最后记录的步**以获取稳态指标（前几步包含编译开销）：

```
step: 20  loss: 11.83506  grad_norm:  9.6669  memory: 48.87GiB(51.44%)  tps: 4,376  tflops: 253.41  mfu: 25.62%
```

### 性能分析

在 recipe 中设置 `config.profiler.trace = KinetoTraceProfiler.Config()`。设置 `KinetoTraceProfiler.Config(freq=...)` 控制捕获哪一步（默认：10）。每个 rank 写入 `{dump_folder}/profiling/traces/iteration_{step}/rank{N}_trace.json.gz`。

```bash
NGPU=8 MODULE=my_graph_configs CONFIG=llama3_8b_profile ./run_train.sh
```

### 内存快照

在 recipe 中设置 `config.profiler.memory_snapshot = MemorySnapshotProfiler.Config()` 以捕获内存快照。快照在每个 `freq` 步触发（默认：10），保存到 `{dump_folder}/profiling/memory_snapshot/`（默认：`./outputs/profiling/memory_snapshot/`）。每个 rank 生成自己的文件：`step_{step:012d}/{N:06d}_step_{step}.pickle`。OOM 时改为转储到 `step_{step:012d}_exit/`。

用 [PyTorch Memory Viz](https://pytorch.org/memory_viz) 工具打开 `.pickle` 文件。

```bash
NGPU=8 MODULE=my_graph_configs CONFIG=llama3_8b_memory_snapshot ./run_train.sh
```

### 逐位确定性护栏

提交任何更改前，先运行逐位确定性测试：
```bash
pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -x
```
该测试验证 aot_fx_trace 路径在多次运行间产生逐位相同的损失和梯度，并与 eager 数值完全匹配。任何破坏此测试的更改必须先调查并修复，再进行其他测试。

### 数值调试

调查数值偏差时，使用 [`.claude/skills/numerics_debugging/SKILL.md`](../../../../.claude/skills/numerics_debugging/SKILL.md) 中的 `numerics_debugging` 技能。

### 异步张量并行（微流水线 TP）

通过 `config.compile.enable_async_tensor_parallel = True` 启用。它使用对称内存（NVLink）将 all-gather + matmul 和 matmul + reduce-scatter 融合为流水线操作。

**何时使用：**
- 启用 TP 且模型隐藏维度较大（TP 切分后 shard_dim >= 1024；例如 llama3 8B dim=4096 在 TP=4 时 shard=1024）。
- 低于此阈值时，流水线分块开销超过重叠收益——Pass 会静默跳过小分片。
- 需要 NVLink 连接的 GPU（H100、A100 NVSwitch 等）。

**示例：**
```bash
NGPU=4 MODULE=my_graph_configs CONFIG=llama3_8b_async_tp4 ./run_train.sh
```

### CUDA Graph Kernel 注解

`insert_kernel_annotations_pass` 在 profiler trace 中用来源 `nn.Module` 路径标记 CUDA graph kernel。使用 `construct_cuda_graph_passes` 构造顺序正确的选择性启用 Pass 列表。profiler 将捕获的注解传给 ``export_chrome_trace``，后者在写入时将其烘焙进 trace，因此自动合并，无需后处理。

要求：`cuda-python` 包和 CUDA toolkit/driver >= 13.1（或 `LD_LIBRARY_PATH` 上的 `cuda-compat >= 13.1`）。这些不可用时 Pass 为空操作。

查看带注解的 trace，请在 https://ui.perfetto.dev 中打开导出的 JSON。Kernel 事件将带有 `module_fqn` 字段，如 `layers.0.attention.wq`。