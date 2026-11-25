# 测试

本目录包含 torchtitan 项目的测试，包括单元测试和集成测试。

## 测试结构

- `unit_tests/cpu/`：无需 GPU 即可运行的单元测试
- `unit_tests/gpu/`：需要 GPU 的测试；多 GPU 测试使用 `multi_gpu` pytest 标记

  两者均镜像源码树：针对 `torchtitan/<path>/<module>.py` 的测试位于 `unit_tests/{cpu,gpu}/<path>/` 下（例如 `models/common/`、`components/data/`、`distributed/flex_shard/`）。`recipes/` 覆盖 `torchtitan_recipes`（覆盖项与 recipe 健全性检查），`scripts/` 覆盖 `scripts/`。针对整个 trainer 的测试保留在顶层。
- `rl/unit_tests/cpu/`：无需 GPU 即可运行的 RL 单元测试
- `rl/unit_tests/gpu/`：需要 GPU 的 RL 测试；多 GPU 测试使用 `multi_gpu` pytest 标记，且每个测试在各自的 `torchrun` 启动下运行
- `integration_tests/`：端到端测试，每个套件一个模块。每个模块将套件的运行配置与引用它们的测试定义放在一起，因此新增测试只需编辑一个文件。
  - `features.py`：torchtitan 特性与可组合性测试
  - `models.py`：模型架构测试
  - `flux.py`：FLUX 模型测试（先训练，再生成图像）
  - `h100.py`：H100 GPU 测试用例
  - `b200.py`：需要 SM100 或 SM103 GPU 的测试用例
  - `rl.py`：RL 端到端套件（`rl`、`rl_h100`、`rl_b200`）
  - `run_tests.py`：所有套件的运行器
  - `hardware_skips/`：硬件特定的跳过列表；参见 [README](integration_tests/hardware_skips/README.md)
- `assets/`：测试使用的测试资产与夹具
  - `losses/`：数值守卫使用的黄金 loss 与梯度范数曲线
  - `tokenizer/`：测试用的分词器配置与词表文件
  - `custom_schedule.csv`：测试用的自定义 PP 调度

## TorchTitan CI 设计

### 集成测试（目标：端到端可组合性）

#### 原则

在 Pull Request 上尽可能使用 Fake PG，以实现快速、广泛的功能覆盖。所有启用的测试在合入前都会运行：兼容 Fake PG 的测试使用一块物理 GPU，标记为 `use_real_pg=True` 的测试使用八块物理 GPU。PR 合入 `main` 后，同一套完整测试会使用 Real PG 运行。定时运行也会使用 Real PG 执行完整套件。

#### 触发节奏

- 1 GPU Fake PG 节奏：就绪的非草稿 Pull Request，无论直接指向 `main` 还是属于 ghstack 栈。
  - `.github/labeler.yml` 文件会自动应用 `ciflow/fake-pg` 标签，PyTorch Probot 会管理一个指向 PR head SHA 的 `ciflow/fake-pg/*` 标签。这使得堆叠的 PR 可以独立于其基础分支运行。该标签会同时启动 Fake-PG 套件和 Real-PG 必需子集。可复用工作流调用方默认只运行 Fake PG。
  - 无法使用 Fake PG 运行的测试（`use_real_pg=True`）会在同一标签触发的独立 `required subset - features` 和 `required subset - models` 任务中使用 Real PG 运行。
- 8 GPU Real PG 节奏：
  - 推送到 `main` 及合入 `main` 会使用 Real PG 运行两个完整套件任务。
  - 每六小时的定时运行和手动触发也会使用 Real PG 运行两个完整套件任务。
  - PR 阶段的可选触发：添加 `ciflow/real-pg` Pull Request 标签会创建 `ciflow/real-pg/*` 标签，并使用 Real PG 运行独立的 `full suite - features` 和 `full suite - models` 任务。
- H100 节奏：携带 `ciflow/h100.8` 标签的可选 Pull Request。该通道始终使用 Real PG；只要标签仍附着，更新和重新打开事件都会重新运行。
- B200 节奏：携带 `ciflow/b200` 标签的可选 Pull Request，以及影响 `main` 上 Kimi K3 的推送。该通道使用 Real PG，目前运行 Kimi K3 多模态覆盖，涵盖 FSDP、TP、EP、PP 和 VPP，外加一项 FSDP+EP 数值回归。

特性测试提供基础设施可组合性的深度。Fake-PG 运行检查特性组合能否配置、转换并完成训练，而 Real-PG 运行还额外覆盖真实集合通信与分布式状态。模型测试提供受支持实现的广度。为清晰起见，它们的定义保持独立。每个 8 GPU Real-PG 套件作为独立的 CI 任务运行，拥有独立的超时。fake-pg 任务将 FLUX 套件与 `features,models` 一起运行；real-pg models 任务在第二次 `run_tests` 调用中运行它，即使第一次失败也会运行，且任一测试套件失败时该任务仍会失败。

### 数值测试（目标：确定性回归覆盖）

部分模型集成测试还通过设置 `golden_numerics_path` 来检查数值。运行器从该文件推导指标和步数，为 A10G Real-PG 执行创建种子检查点，并通过 `loss_compare.py` 运行集成用例。Fake-PG 执行会跳过种子检查点创建，使用其固定的初始化路径。

- A10G 用例在 Pull Request 中使用 Fake PG 在一块物理 GPU 上运行，在合入后、定时运行或由 `ciflow/real-pg` 标签触发时使用 Real PG 在八块物理 A10G 上运行。其黄金路径可使用 `{execution_mode}` 选择 `fake_pg/` 或 `real_pg/` 目录。共享数值用例在两种模式下使用相同配置。
- Fake-PG 黄金值守护 PyTorch FakeProcessGroup 的确定性合成数值契约。它们不验证远程 rank 值或 EP 负载均衡。
- 在该合成契约下，较大的 Fake-PG 梯度范数仍可能是确定性的。不接受非有限梯度：训练会在优化器更新前停止。黄金值能捕获有限合成值的变化；它不能证明模拟梯度在数值上具有代表性。
- 黄金目录标识 PG 模式，文件名标识模型和硬件层级，确切的并行方案记录在文件头中。
- Qwen3.5 MoE FSDP 4 x TP 2、EP 4 没有数值黄金值。其 A10G Real-PG 结果不是逐位确定性的，因此该用例仅提供端到端覆盖。
- DeepSeek V4 FSDP 2 x TP 2、EP 2 仅提供端到端覆盖，没有数值黄金值。它仅在真实 PG 上运行（`use_real_pg=True`）：在 Fake PG 下，其序列并行集合通信返回的激活值会别名其输入，破坏为反向保存的张量，并在第 1 步导致 grad_norm 爆炸。同一配置在真实的 4-GPU PG 上可正常训练。

| A10G 模型 | 拓扑（Fake-PG 与 Real-PG） |
| --- | --- |
| Llama 3 | FSDP 2 x TP 2 x CP 2 |
| Llama 3 SFT | FSDP 2 |
| DeepSeek V3 | FSDP 8, EP 8 |
| DeepSeek V4 | FSDP 2 x TP 2, EP 2（仅 Real-PG） |
| GPT-OSS | FSDP 4 x TP 2, EP 4 |
| Qwen3 | FSDP 2 x TP 2 x CP 2, EP 8 |
| Muse Glimmer text | FSDP 8 |
| Qwen3.5 MoE multimodal | FSDP 4 x TP 2, EP 4 |
| Kimi K2.5 DistMuon | FSDP 8, EP 8 |

Kimi K2.5 DistMuon FSDP+EP 和 Kimi K2.7 DistMuon PP+FSDP+EP 均在 A10G 上运行。K2.5 多模态反向使用双三次上采样，其 CUDA 反向没有确定性实现，因此 FSDP+EP 用例不携带数值黄金值。如需手动比较，`loss_compare.py` 可以使用模型等效的 AdamW 配置创建仅模型的种子检查点，而实测运行继续使用 DistMuon。

额外的 A10G 仅 Real-PG 用例覆盖 CP 和流水线通信：

| A10G 模型 | 流水线并行拓扑 |
| --- | --- |
| DeepSeek V3 | FSDP 2 x CP 2 x PP 2, EP 4, Interleaved1F1B |
| Llama 3 | FSDP 2 x TP 2 x PP 2, 1F1B |
| GPT-OSS | FSDP 2 x CP 2 x PP 2, EP 4, Interleaved1F1B |
| Kimi K2.7 DistMuon | FSDP 2 x PP 2, EP 2, Interleaved1F1B |

在 Pull Request 上，Fake-PG 通道和 `real_pg_required` Real-PG 范围无重叠地划分启用的 A10G 测试。合入后和定时的 Real-PG 通道运行完整的选定套件。如果选定的配置使用了检查点、流水线并行、显式的非 Fake 通信后端或其他已知不兼容项而未设置 `use_real_pg=True`，则验证失败。硬件由套件编码：`features` 和 `models` 在 A10G 通道中运行，而 `h100` 和 `b200` 仅在其硬件特定的工作流中使用 Real PG 运行。

### 单元测试（目标：模块功能）

- CPU 与 GPU 需求由 `unit_tests/cpu/` 和 `unit_tests/gpu/` 目录编码。
- 需要多块物理设备的 GPU 测试使用 `multi_gpu` pytest 标记。1-GPU 通道选择 `not multi_gpu`，而多 GPU 通道从同一 GPU 目录中选择 `multi_gpu`。
- RL 单元测试在独立的 1-GPU CUDA 通道中使用 RL 镜像运行，该镜像包含 vLLM、Monarch、TorchStore 及示例依赖。多 GPU RL 一致性测试和 RL 训练循环在 RL 集成通道中运行。

## 运行测试

### 前置条件

确保已安装所有开发依赖：

```bash
pip install -r requirements-dev.txt
pip install -r requirements.txt
```

### 运行集成测试

运行集成测试：

```bash
python -m tests.integration_tests.run_tests <output_dir> [--test_suite TEST_SUITE[,TEST_SUITE...]] [--execution_mode {fake_pg,real_pg}] [--test_scope {all,real_pg_required}] [--test_name TEST_NAME] [--ngpu NGPU]
```

参数：
- `output_dir`：（必需）存储测试输出的目录
- `--test_suite`：（可选）要运行的测试套件，逗号分隔（默认："features"）
- `--execution_mode`：（可选）使用 Fake PG 或 Real PG 运行（默认：`real_pg`）
- `--test_scope`：（可选）运行所有选定测试，或仅运行标记为 `use_real_pg=True` 的测试（默认：`all`）
- `--test_name`：（可选）按名称运行特定测试（默认："all"）
- `--ngpu`：（可选）用于测试的 GPU 数量（默认：8）
- `--export-numerics`：（可选）导出数值测试结果，而不是与其 `golden_numerics_path` 文件比较
- `--hf_assets_path`：（可选）导出到配置中的 HF 检查点，作为 `TORCHTITAN_TEST_HF_ASSETS_PATH`（供需要真实权重的 RL 测试使用）

每个测试都会命名其运行的完整配置，每次运行一个。默认情况下，每个配置通过 `run_train.sh` 运行 `torchtitan.train`，该脚本接收 `MODULE`、`CONFIG` 和 `ENTRYPOINT` 环境变量。测试可以为每个配置设置 `entrypoints` 以启动另一个模块（例如 Flux 图像生成）。RL 测试使用 `torchtitan.rl.train`，它通过 Monarch 放置自己的进程，因此运行器使用 `python -m` 启动它，并顺序运行 RL 套件。套件配置位于套件模块本身；可复用的模型调试配置位于 [torchtitan_recipes/tests](../torchtitan_recipes/tests/)。

示例：
```bash
# Run all feature integration tests (features is the default suite)
python -m tests.integration_tests.run_tests test_output

# Run the complete A10G matrix with Fake PG on one physical GPU
python -m tests.integration_tests.run_tests test_output --test_suite features,models --execution_mode fake_pg --ngpu 1

# Run the complete A10G matrix with real process groups
python -m tests.integration_tests.run_tests test_output --test_suite features,models --execution_mode real_pg --ngpu 8

# Run only cases that explicitly require real process groups
python -m tests.integration_tests.run_tests test_output --test_suite features,models --execution_mode real_pg --test_scope real_pg_required --ngpu 8

# Run H100-only cases with real process groups
python -m tests.integration_tests.run_tests test_output --test_suite h100 --execution_mode real_pg --ngpu 8

# Run B200-only cases with real process groups
python -m tests.integration_tests.run_tests test_output --test_suite b200 --execution_mode real_pg --ngpu 8

# Run the RL end-to-end suite (requires the RL dependencies)
python -m tests.integration_tests.run_tests test_output --test_suite rl --ngpu 8
```

### 运行单元测试

仅运行单元测试：

```bash
pytest -s tests/unit_tests/cpu/

# Single-GPU tests
pytest -s tests/unit_tests/gpu/ -m "not multi_gpu"

# Multi-GPU tests
pytest -s tests/unit_tests/gpu/ -m multi_gpu

# RL unit tests (require the RL dependencies)
pytest -s tests/rl/unit_tests/cpu/
pytest -s tests/rl/unit_tests/gpu/ -m "not multi_gpu"

# RL multi-GPU unit tests run one torchrun launch per test, for example
torchrun --nproc-per-node=2 -m pytest tests/rl/unit_tests/gpu/test_bitwise_parity.py::TestBitwiseParityVarlen
```

RL CPU 和 GPU 单元工作流（`unit_test_cpu_rl.yaml`、`unit_test_gpu_rl.yaml`）在 `main` 推送、每 12 小时以及带有 `ciflow/rl` 标签的 PR 上运行，该标签会自动添加到触及 RL 代码的 PR（参见 `.github/labeler.yml`）。当 PR 更改 `.ci/docker/` 时，使用现有的 `ciflow/docker` 标签构建其镜像；镜像可用后重新运行 RL 单元任务。

H100/B200 RL 工作流（`integration_test_rl_h100.yaml`、`integration_test_rl_b200.yaml`）在独立 GPU 检查和完整 GRPO 集成之间共享环境设置：

| 触发方式 | H100 | B200 |
| --- | --- | --- |
| PR 标签（分别为 `ciflow/rl/*` 或 `ciflow/b200/*`） | 独立检查 | 独立检查 |
| 推送到 `main` | 独立检查 | 不运行 |
| 每天 UTC 00:00 在 `main` 上 | 独立检查 + GRPO | 独立检查 + GRPO |
| 每天 UTC 12:00 在 `main` 上 | 独立检查 | 不运行 |
| 手动触发 | 独立检查 + GRPO | 独立检查 + GRPO |

这些工作流仍会预留八 GPU 的 H100/B200 运行器。其 GRPO 套件覆盖 Kimi K3 和 DeepSeek V3，以及 B200 上的 Dist-MoE；这些配置每天自动检查一次，而不是在 PR 标签或 `main` 推送时检查。

手动运行需要仓库写入权限。使用 **Run workflow** 选择上游分支，或使用 CLI 触发上游分支或标签：

```bash
gh workflow run integration_test_rl_h100.yaml --repo pytorch/torchtitan --ref BRANCH_OR_TAG
```

对于 Fork PR，维护者必须使用指向所需 PR 提交的上游分支或标签，例如其 `ciflow/rl/*` 或 `ciflow/b200/*` 标签；Fork 分支在上游分支选择器中不可用。在 `main` 上的手动运行不会取消定时运行；在同一非 `main` ref 上的重复运行会取消上一次运行。

### 运行特定单元测试文件

运行特定测试文件：

```bash
pytest -s tests/unit_tests/cpu/config/test_config_loader.py
```

### 运行单元测试中的特定测试函数

运行特定测试函数：

```bash
pytest -s tests/unit_tests/cpu/config/test_config_loader.py::test_operational_overrides
```