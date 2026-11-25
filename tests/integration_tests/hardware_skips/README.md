# 硬件特定的集成测试跳过

本目录下的 `skipped_tests.toml` 列出了各硬件后端上跳过的集成测试。`tests/integration_tests/run_tests.py` 会读取它：匹配的测试在启动前被剔除，并记录跳过日志。

## 文件格式

每个硬件后端一个顶层键，值为该后端跳过的测试列表：

```toml
rocm = [
  "sdc_replay_accelerator_graph",
  "fsdp_pp_deferred_gradient_reduction",
  "deepseek_v3_fused_mla_swiglu_fsdp+ep",
  "dist_gemm",   # symmetric memory is CUDA-only
]
```

- 键与 `utils.device_type` 对应（例如 `xpu`）。
- `cuda` 和 `rocm` 是特例：ROCm 也报告 `device_type == "cuda"`，因此用 `torch.version.hip` 区分两者。
- 值为 `IntegrationTestDefinition` 条目的 `test_name` 列表。同一个名称可列在多个后端下。
- 原因写在条目的行尾注释中；值若不是字符串数组，会在任何测试启动前抛出 `ValueError`。
- 本目录存放 `skipped_tests.toml`；树外硬件按下文所述注册自己的文件。

## 树外设备仓库如何注册跳过

树外硬件通过 `register_hardware_skips` 注册跳过。设备仓库是为 torchtitan 提供额外设备后端的另一个仓库。`register_hardware_skips` 接收 TOML 文件路径。有两种方式，按优先级排列：

### 1. 入口点（自动注册）

设备仓库在其 `pyproject.toml` 中添加：

```toml
[project.entry-points."torchtitan.integration_tests.hardware_skips"]
example_hardware = "example_hardware.tests:register_skips"
```

还需添加一个 Python 文件，用其维护的 TOML 文件调用注册函数：

```python
# example_hardware/tests.py
from tests.integration_tests.hardware_skips import register_hardware_skips


def register_skips() -> None:
    register_hardware_skips(toml_file_path)
```

### 2. 显式调用

用你维护的 TOML 文件显式调用 `register_hardware_skips`。仓库内示例见 `torchtitan/experiments/graph_trainer/tests/integration_tests.py`：

```python
# torchtitan/experiments/graph_trainer/tests/integration_tests.py
register_hardware_skips(
    Path(__file__).parent / "hardware_skips" / "skipped_tests.toml"
)
run_tests(args, test_list)
```

## 跳过应放在哪里

| 机制 | 用途 | 位置 |
|---|---|---|
| `hardware_skips/skipped_tests.toml` | 永久的、后端范围的跳过 | 本目录 |
| `disabled=True` | 非硬件耦合导致的跳过，以及临时一次性跳过 | 定义处 |
| `--exclude a,b,c` | 单次调用跳过少量用例 | 命令行 |

仓库内已在某个 CI 任务剔除少量用例时使用 `--exclude`：
[scripts/ci/pytorch_ci_test_runner.sh:37](../../../scripts/ci/pytorch_ci_test_runner.sh#L37)、
[.github/scripts/validate_release_gpu.sh:382](../../../.github/scripts/validate_release_gpu.sh#L382)、
[.github/scripts/run_8xgpu_integration_tests.sh:81](../../../.github/scripts/run_8xgpu_integration_tests.sh#L81)。