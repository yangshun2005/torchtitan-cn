# Autoresearch

Autoresearch 是一个自主实验循环：LLM 智能体针对训练图迭代 GPU 内核 / 图通道优化，在保持与 eager 参考实现逐位相同数值的前提下提升速度。循环无限运行，直到手动停止。

本文件是智能体的操作手册。标记为 **[SETUP]** 的章节在循环开始前按每次运行填写；其余部分是通用的 harness 约定。

## [SETUP] 本次运行

在开始循环前描述本次运行：

- **目标模型 / 配置**：例如 `Llama3 8B FSDP=4 TP=2 bs=1 on 8×H100`。
- **起始图**：例如空的 `construct_default_graph_passes`（原始 aten 图）vs. 生产通道集（已应用 cleanup + bucketing + 区域 Inductor + CUDA graphs）。
- **脚手架级别**：智能体可使用 {精选 IDEAS、仓库内参考实现、在线调研} 中的哪些。相应收紧下方的 *阅读范围*。
- **目标**：衡量什么（例如"在生产图之上寻找进一步收益"）。

## 阅读范围（非常重要）

### 你可以阅读

- `torchtitan/experiments/graph_trainer/passes.py` —— **你唯一修改的文件**。添加自定义通道函数，并在 `construct_default_graph_passes` 或 `compile_time_passes` 中注册。
- `torchtitan/experiments/graph_trainer/trainer.py` —— GraphTrainer（只读）。
- `torchtitan/experiments/graph_trainer/make_fx_tracer.py` —— make_fx 追踪（只读）。
- `torchtitan/experiments/graph_trainer/compile.py` —— 编译入口（只读）。
- `torchtitan/experiments/graph_trainer/<model>/` —— 模型的 graph_trainer 胶水代码（只读）。
- `torchtitan/models/<model>/`、`torchtitan/models/common/` —— 模型架构（RMSNorm、Attention、RoPE、FF、Decoder）—— 只读。
- `torchtitan/experiments/graph_trainer/autoresearch/ideas.md`、`torchtitan/experiments/graph_trainer/autoresearch/learnings.md`、`torchtitan/experiments/graph_trainer/autoresearch/experiment_log.md` —— 你的活文档。
- `torchtitan/experiments/graph_trainer/autoresearch/scripts/` —— 你拥有的可复用工具；按需创建/修改。
- 你自己生成的 FX 图转储（例如通过临时的 `gm.print_readable()` 通道）。

### 你不得阅读

不要阅读数值测试源码（见下文）。

不确定某个源码是否属于可读范围时，倾向于**不读**。

### 你不得查看 git 历史

- 不要运行 `git log`、`git log -p`、`git show <commit>`、`git diff <commit>` 或 `git blame`。历史提交的 diffstat 和标题行都引用了我们希望你能自主重新发现的优化技术。
- `git status` 和 `git diff` / `git diff --staged`（不带 commit 引用，仅工作区）可以。
- 要查找*你*在本次运行中创建的提交，查看 `torchtitan/experiments/graph_trainer/autoresearch/results.tsv` —— 第二列记录了每个 `keep` 实验的提交哈希。

## 约束

**只修改** `torchtitan/experiments/graph_trainer/passes.py`。

**不要**修改 `trainer.py`、`make_fx_tracer.py`、模型代码，或添加依赖。

**目标：在保持逐位相同数值的前提下最小化训练步时间。**

**内存**是软约束 —— 为获得有意义的加速，允许适度增加。

**简洁性**：其他条件相同时，越简单越好。

## 基准测试

```bash
bash torchtitan/experiments/graph_trainer/autoresearch/scripts/run_benchmark.sh
```

从最后一步提取稳态指标：

```bash
grep "step:" run.log | tail -1
```

输出格式：
```
step: 20  loss: 8.12345  grad_norm: 1.2345  memory: 12.34GiB(25.00%)  tps: 12,345  tflops: 123.45  mfu: 12.34%
```

性能分析 trace：在基准测试配方中设置 `config.profiler.trace = KinetoTraceProfiler.Config()`。

**图检查**：在你的通道中添加临时的 `gm.print_readable()` 调用，运行 1-2 步，检查输出。基准测试前移除调试打印。

## [SETUP] 验证数值

数值门禁通过逐位对比 eager 测试，将优化后的图与 eager 参考实现比较，例如：

```bash
pytest torchtitan/experiments/graph_trainer/tests/test_bitwise_deterministic.py -k "TestLlama3BitwiseDeterministic and aot_fx_trace_vs_eager" -x > numerics.log 2>&1
```

（将 `TestLlama3...` 替换为模型的测试用例。）任何失败都计为 `crash`，无论性能提升如何。

**不要阅读数值测试文件的源码** —— 它导入的符号名称会泄露先前的优化技术。运行它，失败时检查 numerics.log 中的断言消息，仅此而已。

## 记录结果

追加到 `torchtitan/experiments/graph_trainer/autoresearch/results.tsv`（制表符分隔）。8 列：

```
timestamp	commit	tps	memory_gib	status	mfu_pct	wall_time_s	description
```

1. ISO 8601 时间戳
2. Git 提交哈希（7 字符）。`discard`/`crash` 用 `xxxxxxx`。
3. 每秒 token 数。崩溃时为 0。
4. 峰值内存 GiB（.1f）。崩溃时为 0.0。
5. 状态：`keep`、`discard` 或 `crash`
6. MFU 百分比。崩溃时为 0.0。
7. 实际耗时（秒）。崩溃时为 0。
8. 简短描述。

## 实验循环

永远循环：

1. 重新阅读 `ideas.md`、`learnings.md` 和 `experiment_log.md`。选择下一个想法。
2. **将实现委派给子智能体。** 启动一个子智能体，用清晰的提示描述在 `passes.py` 中要实现什么。子智能体完成所有代码修改、运行基准测试并验证数值。这能保持主循环上下文干净。子智能体提示应包含：
   - 尝试什么优化以及为什么。
   - 上述阅读范围限制。逐字传递 —— 子智能体必须遵守相同规则。
   - 基准测试命令：`bash torchtitan/experiments/graph_trainer/autoresearch/scripts/run_benchmark.sh`
   - 如何提取结果：`grep "step:" run.log | tail -1`
   - 数值检查（*验证数值* 章节中的命令）。
   - 空 grep 或数值失败 = `crash`。
3. 读取子智能体的结果。确定状态：tps 提升 + 数值通过 → `keep`；无提升 → `discard`；崩溃或数值失败 → `crash`。记录到 TSV。
4. 向 `experiment_log.md` 追加条目。
5. 若 `keep`：提交 `passes.py`、`results.tsv`、`experiment_log.md`、`ideas.md`、`learnings.md`。消息：`[autoresearch] <简短描述>`。
6. 若 `discard` 或 `crash`：仅还原 `passes.py`（`git checkout -- torchtitan/experiments/graph_trainer/passes.py`），然后提交更新后的跟踪文件（`results.tsv`、`experiment_log.md`、`ideas.md`、`learnings.md`），以便后续迭代记住哪些没成功。消息：`[autoresearch] <简短描述> (<状态>)`。

**基准测试方差**：TPS 波动约 1-3%。若差值在噪声范围内，重跑确认。

**崩溃**：修复简单 bug 后重跑。若根本性损坏，跳过并继续。

**永不停止**：无限工作，直到手动中断。若想法用尽，做性能分析、研究图、尝试不同方法。