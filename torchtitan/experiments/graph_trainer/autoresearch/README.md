# Autoresearch Harness

用于 TorchTitan `graph_trainer` 中图 pass / kernel 优化的自主实验循环。LLM agent 反复编辑单个文件（`torchtitan/experiments/graph_trainer/passes.py`），对每次改动做基准测试，仅当训练步时间改善**且**数值与 eager 参考逐位一致时才保留该改动。循环无限运行，直到手动停止。

Harness 位于 `torchtitan/experiments/graph_trainer/autoresearch/`。本 README 中的文件路径均相对于该目录；命令在仓库根目录执行。

## 文件

| 文件 | 负责人 | 用途 |
|---|---|---|
| `autoresearch.md` | 实验者 + agent | agent 的操作手册。开始运行前填写 `[SETUP]` 部分。 |
| `ideas.md` | 实验者 + agent | 优化想法。可预置精选方向，或留空以完全自主。 |
| `experiment_log.md` | agent | 仅追加的日志：每次实验的想法 / 改动 / 结果 / 分析 / 经验教训。 |
| `learnings.md` | agent | 提炼后的跨实验经验。 |
| `results.tsv` | agent | 机器可读的性能跟踪（每次实验一行）。 |
| `scripts/run_benchmark.sh` | 实验者 | 基准测试命令——按你的运行修改模型/配置/并行度。 |

## 开始运行

1. 按你的模型、配置和并行度编辑 `scripts/run_benchmark.sh`。
2. 填写 `autoresearch.md` 的 `[SETUP]` 部分：目标、起始图、脚手架级别（精选想法 / 参考访问 / 网络）、阅读范围限制，以及数值检查命令。
3. 可选：用精选优化方向预置 `ideas.md`。
4. 将基线记录为 `results.tsv` 的第一行和 `experiment_log.md` 的第一条。
5. 让 agent 读取 `autoresearch.md`，然后启动循环。

## 设计特性

- **异步**：共享文件是人类与 agent 之间唯一的通信渠道；双方无需同时在线。
- **可审计**：每次实验都记录完整上下文，提交树保留所有代码状态（包括失败）。
- **安全**：数值正确性是硬性门槛。失败的实验只回退 `passes.py`；当前最优状态始终是 HEAD。
- **可累积**：每次 `keep` 都建立在之前所有 `keep` 之上。