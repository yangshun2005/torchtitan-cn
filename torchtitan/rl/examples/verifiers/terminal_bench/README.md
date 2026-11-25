# 使用 Verifiers 和 TitanRL 的 Terminal-Bench

本示例使用 TitanRL 在 AI2 的 TMax 任务上训练终端智能体，并在 89 个 Terminal-Bench 2.1 任务上验证。每次 rollout 是对一个任务的一次尝试：Harbor 的 Terminus-2 智能体在任务的 Docker 容器中工作，每次模型调用都发往 TitanRL 的生成器，任务自带的测试负责评分。Verifiers 通过 [Verifiers 桥接](../README.md) 运行智能体和评分。

> **玩具沙箱。** 本地 Docker 让本示例自包含；它不是用于真实训练的沙箱。容器在控制器主机上运行不受信任的任务代码，且默认共享其网络。真实训练请接入成熟的沙箱作为 Verifiers 运行时。不要使用 `SubprocessConfig`：它会在主机上运行智能体。

## 工作原理

每次 rollout 走 Verifiers README 中[集成流程](../README.md#integration-flow)的路径。此处：

- **智能体。** 环境服务器根据任务的 `docker_image` 启动 Docker 容器，并在其中运行 Terminus-2（[`harness.py`](./harness.py)）。智能体向 tmux shell 输入命令并读回屏幕。每个智能体轮次是一次模型调用。
- **评分。** 智能体停止后，Verifiers 在容器中运行任务的 `tests/test.sh`，并读取它写出的奖励文件。

验证在 Terminal-Bench 2.1 上走相同路径。验证随训练启动并与其并行运行（`ValidationConfig`）。

| 内容 | 位置 |
| --- | --- |
| 配方：模型、并行、生成器、训练循环，以及 Verifiers 设置（数据集、harness、Docker 运行时、超时） | `torchtitan_recipes/rl/verifiers_terminal_bench.py` |
| Terminus-2 harness 及其在容器中运行的程序，从 Verifiers 0.3.1 复制，以在历史中暴露推理并支持摘要 | [`harness.py`](./harness.py)、[`terminus_harness.py`](./terminus_harness.py) |
| Harbor 任务集，在此声明以便环境服务器 worker 导入 | [`taskset.py`](./taskset.py) |
| 离线 TMax 到 Harbor 任务转换 | [`prepare_tmax.py`](./prepare_tmax.py) |
| 桥接：任务数据集、环境服务器、生成服务器、trace 到轮次 | [Verifiers 集成](../README.md) |
| CPU 测试 | `tests/rl/unit_tests/cpu/test_verifiers_terminal_bench*.py` |

## 运行

在同一个 Python 3.12 环境中安装 TorchTitan RL 依赖和本目录的 `requirements.txt`，并让用户在控制器主机上运行 Docker。准备训练任务（见[数据集](#数据集)），然后：

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.verifiers_terminal_bench \
  --config rl_grpo_qwen35_9b_terminal_bench
```

9B 配方将 `hf_assets_path` 设为 `./assets/hf/Qwen3.5-9B`，35B 配方设为 `./assets/hf/Qwen3.5-35B-A3B`。要使用其他目录，请修改 `torchtitan_recipes/rl/verifiers_terminal_bench.py` 中的该赋值。该路径没有 CLI 标志。

| 配置 | 模型 | GPU |
| --- | --- | --- |
| `rl_grpo_qwen35_9b_terminal_bench` | Qwen3.5-9B | 8 训练，8 生成器 x 1 |
| `rl_grpo_qwen35_35b_a3b_terminal_bench` | Qwen3.5-35B-A3B | 8 训练，2 生成器 x 4 |

`torchtitan.rl.train` 将所有进程放在本地主机上，因此按原样两个配方都需要 16-GPU 主机；目前没有多主机启动器示例。35B-A3B 配方尚未端到端运行。

## 数据集

Verifiers 从 `~/.cache/harbor/org_name_ref` 加载 Harbor 数据集 `org/name@ref`，若该目录不存在则运行 `harbor download`。

| 角色 | 数据集 id | 来源 |
| --- | --- | --- |
| 训练 | `local/tmax@v1` | AI2 的 [TMax](https://huggingface.co/datasets/allenai/tmax-15k-open-instruct)，由 `prepare_tmax.py` 转换 |
| 验证 | `terminal-bench/terminal-bench-2-1` | 来自 Harbor Hub 的 Terminal-Bench 2.1 |

`local/tmax@v1` 不在 Harbor Hub 上。下载 Hugging Face 数据集并转换一次：

```bash
python -m torchtitan.rl.examples.verifiers.terminal_bench.prepare_tmax \
  --parquet tmax/data --task-data tmax/task-data.tar.gz \
  --out ~/.cache/harbor/local_tmax_v1
```

- 每个任务必须声明可拉取的 `[environment].docker_image`。Verifiers 不构建 Dockerfile，仅含 Dockerfile 的单个任务会导致数据集加载失败。`prepare_tmax.py` 写入每个 TMax 任务已发布的镜像。
- 任务自带的超时被忽略；配方文件中的超时生效。
- 验证 id 没有 `@ref`，因此使用缓存中的任意修订版。要获得可比较的数字，请固定 ref；Terminal-Bench 2.0 和 2.1 不可互换。
- 无法访问 Harbor Hub 时，用导出的任务树预填充 `~/.cache/harbor`。
- 要使用其他数据集，请修改配方文件中传给 `_terminal_bench_rollouter_config` 的 id。训练和验证数据集必须不同。

## 如果所有奖励都是 0

- **从未运行的评分器得 0 分。** Verifiers 忽略 `tests/test.sh` 的退出状态，将缺失或不可读的奖励文件记为 0，与任务失败相同。Terminal-Bench 2.1 测试脚本在运行时安装测试工具（`apt-get`、`uv`、pytest），因此评分时容器需要出站网络。Verifiers 的 Docker 运行时默认网络策略使用主机网络，因此控制器主机需要该访问权限。
- **连续 10 个不可训练批次后运行失败。** 所有样本获得相同奖励的组会被丢弃（`TrainingSampleBuilder.drop_zero_std_reward_groups`）。批处理器在连续出现一批此类组后发出警告，并在 10 次后抛出 "consecutive untrainable batches"。长时间运行前，请检查几个任务的评分，并确认模型能解决其中一些。

## 已知问题

- Verifiers 用普通 `tar -x` 解压每个任务的 `tests/`，这会恢复主机文件所有权，并在主机 ID 未映射的无根容器中失败。