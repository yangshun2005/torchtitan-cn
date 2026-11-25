# 为 torchtitan 做贡献
我们希望让为本项目做贡献尽可能简单透明。贡献应遵循下方的[贡献指南](#contributing-guidelines)。

### 环境准备
```
pip install -r requirements.txt -r requirements-dev.txt
```

### Pull Request
我们积极欢迎你的 Pull Request。

1. Fork 仓库，并从 `main` 创建你的分支。
2. 每个 PR 只解决一个问题，并尽可能小。
3. 如果改动了 API，请更新文档。
4. 确保测试套件通过。
5. 确保代码通过 lint 检查（`pre-commit run --all-files`）。
6. 填写 PR 模板：用简单示例说明具体问题、改了什么、数值/性能证据以及测试情况。评审者需要被说服，而不只是被告知。
7. 自己撰写"Summary (human-written)"部分。参见 [AI 辅助贡献](#ai-assisted-contributions)。
8. 如果尚未完成，请签署贡献者许可协议（"CLA"）。

领域专家会根据 [.github/area_reviewers.yml](.github/area_reviewers.yml) 自动被邀请为评审者。维护者可以通过评论 `@claude review` 请求 AI 评审；它遵循 [.claude/skills/pr-review](.claude/skills/pr-review/SKILL.md)。

### AI 辅助贡献
我们遵循 PyTorch 的 [AI 政策](https://github.com/pytorch/pytorch/blob/main/AI_POLICY.md)。你可以使用 AI 工具，但必须由理解该改动的人负责。

- 自己撰写 PR 的"Summary (human-written)"。几句话即可：问题是什么、为什么采用这种方法、你自己检查了什么。摘要为空或由 AI 撰写的 PR 可能被关闭或退回草稿状态。请在 GitHub 的 PR 描述中撰写摘要。合并的提交信息取自 PR 描述，而非你本地的提交信息。如果你的工具在每次推送时用提交信息重写 PR 描述（不带 `--keep-body` 的 stack-pr、`ghstack -u`），请关闭该行为以保留摘要。
- 其他部分可以由 AI 起草。你要对其中每一项声明负责，因此请对照代码和日志核对每个数字和陈述。
- 在请求评审前先阅读并简化代码。在此之前将 PR 保持为草稿状态。
- 不要用 AI 的原始输出回复评审者。如果引用 AI 输出，请放在代码块或引用块中，并附上你自己的看法。
- 我们不接受完全由自主智能体提交的 PR。

### 贡献者许可协议（"CLA"）
为了接受你的 Pull Request，我们需要你提交 CLA。你只需完成一次，即可参与 Meta 的任何开源项目。

在此完成你的 CLA：<https://code.facebook.com/cla>

### Issue
我们使用 GitHub Issue 跟踪公开 bug。请确保描述清晰，并包含足够的复现步骤。

Meta 设有[漏洞赏金计划](https://www.facebook.com/whitehat/)，用于安全漏洞的安全披露。此类情况请按照该页面所述的流程处理，不要提交公开 Issue。

### 许可证
向 `torchtitan` 做贡献即表示你同意你的贡献将按照本源码树根目录下 LICENSE 文件的条款进行许可。

---

## 贡献指南

注意：为加速对 `torchtitan` 的贡献和创新，我们新增了一个实验性文件夹 [`torchtitan/experiments`](torchtitan/experiments/)，它有自己的[贡献指南](torchtitan/experiments/README.md#contributing-guidelines)。以下内容针对 `torchtitan` 的核心部分。

### 贡献原则

- 采用 PyTorch 原生的训练技术。
  - 该技术应对分布式训练具有普遍意义。
  - 中等及以上复杂度的技术应放在合适的仓库中（例如新的并行方式放在 pytorch/pytorch，新的数据加载器放在 pytorch/data），而不是 `torchtitan`。
  - `torchtitan` 主分支应尽量少依赖非 PyTorch 库。依赖外部库的有趣模型/技术可以在 `experiments` 文件夹或 `torchtitan` 的 Fork 中演示。
- 力求对模型的代码改动最小（最好为零）。对于 `torchtitan` 中的 Llama 模型，如果必须做合理的模型改动：
  - 改动后仍应能正确加载原始 checkpoint。
  - 记录改动原因，类似 [PIPELINE_PARALLEL.md](torchtitan/distributed/PIPELINE_PARALLEL.md) 中的模型说明。
- 保持代码模块化，尤其是 [train.py](torchtitan/train.py)，以便仍能轻松复制粘贴到最小代码示例中。如有必要，创建独立的函数/文件。
- 训练行为由 Python recipe 配置，而非 `--section.option` 标志。改变模型的开关放在模型配置 dataclass 中。属于某个组件的开关放在该组件的配置中，没有其他归属的开关在与维护者确认后放在 [configs.py](torchtitan/config/configs.py)。参见[配置文档](torchtitan/config/README.md)。

### 价值证明

贡献者有责任证明改动的合理性。要求包括但不限于：

#### Loss

- 如果改动不影响计算结果，在固定随机种子（`config.debug.seed`）和确定性算法（`config.debug.deterministic`）下，改动前后的 loss 应完全一致。激活检查点（activation checkpointing）就是一个例子。
- 如果改动预期会影响计算结果，应通过在具有代表性的数据集上进行端到端训练来验证 loss 收敛（例如在 C4 数据集上训练 Llama 3 模型）。请参考 [debugging.md](docs/debugging.md#validating-a-distributed-training-technique) 中的推荐做法。

#### 性能
- 日志中可获取的内存和 TPS / MFU 应符合预期。
- 需要注意的是，性能预期因情况而异。例如，某些针对内存优化的技术可能导致吞吐量回退但仍可接受（如激活检查点）。同样，贡献者有责任证明该特性的合理性，无论是通过达到假设的性能，还是与现有的知名实现进行比较等。
- 如有必要，在跨多节点的任务上验证数字（例如在 64 块 GPU 上）。如果资源受限，请联系 `torchtitan` 团队寻求帮助。
- 在适当情况下，应展示 profile trace 和/或内存快照以证明有效性。

### 最佳实践

在适当情况下，应考虑：

- 添加守护真实不变量的 CPU/GPU 单元/集成测试。CI 容量有限，因此优先扩展现有测试或集成配置，而非新增，并说明新测试能捕获哪种回归。
  - 添加单元测试：放在 [tests](tests/) 文件夹中，并遵循现有测试文件。
  - 添加 GPU 集成测试：在 [tests/integration_tests](tests/integration_tests/) 中对应的套件模块里添加一个配置函数和一个引用它的 `IntegrationTestDefinition`。可复用的模型调试配置位于 [torchtitan_recipes/tests/models](torchtitan_recipes/tests/models/)。
- 更新 [README](README.md) 并在代码旁记录该特性：用法写在包的 `README.md` 中，设计说明写在大写 `.md` 文件中，类似 [MXFP8](torchtitan/quantization/mxfp8/README.md)。
- 新模型代码遵循张量形状后缀命名约定（例如 `x_BLD`、`q_BLHK`、`out_THV`），并在每个模块中附上图例注释，如 [attention.py](torchtitan/models/common/attention/attention.py)。大写后缀命名逻辑张量维度（而非分片布局），作用域为单个文件。
- 在 [benchmarks](docs/benchmarks) 文件夹中添加包含基准测试结果的新文件。
- 为当前无法解决的问题创建 GitHub Issue。
- 在 [PyTorch 论坛](https://discuss.pytorch.org/c/distributed/torchtitan/44)上发帖并链接到该帖。