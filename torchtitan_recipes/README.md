# TorchTitan 配方

本包包含完整、可运行的训练配置。`torchtitan.models` 下的核心模型包包含架构定义和模型变体，但不选择数据集、优化器、并行方式或集群路径。

公开配方仅限于项目在训练、收敛、性能或维护的功能工作流中实际验证过的配置。仅用于 CI 的配置位于 `torchtitan_recipes.tests` 下，不推荐用于真实训练。

本包按归属组织：

- `models/` 包含核心模型的已验证配方。
- `sft/` 包含监督微调配方，每个数据集一个模块。
- `graph_trainer/`、`torchft/` 和 `rl/` 对应其所属的实验或训练子系统。
- `overrides/` 包含可选的组件实现。
- `tests/` 包含仅用于 CI 的模型测试夹具（集成测试套件配置与其测试定义一起位于仓库的 `tests/integration_tests/` 中）。

| 模块 | 范围 | 验证参考 |
| --- | --- | --- |
| `torchtitan_recipes.models.llama3` | Llama 3 预训练 | `torchtitan/models/llama3/README.md`、`docs/benchmarks/` |
| `torchtitan_recipes.models.deepseek_v3` | DeepSeek V3 预训练 | `torchtitan/models/deepseek_v3/README.md` |
| `torchtitan_recipes.models.qwen3` | Qwen3 预训练和 SFT | `torchtitan/models/qwen3/README.md` |
| `torchtitan_recipes.models.flux` | Flux 预训练 | `torchtitan/models/flux/README.md` |
| `torchtitan_recipes.models.muse_glimmer` | Muse Glimmer 预训练 | `torchtitan/models/muse_glimmer/README.md` |
| `torchtitan_recipes.models.muse_spark` | Muse Spark 预训练 | `torchtitan/models/muse_spark/README.md` |
| `torchtitan_recipes.graph_trainer.llama3` | GraphTrainer Llama 3 | `torchtitan/experiments/graph_trainer/README.md` |
| `torchtitan_recipes.graph_trainer.qwen3` | GraphTrainer Qwen3 | `torchtitan/experiments/graph_trainer/README.md` |
| `torchtitan_recipes.graph_trainer.deepseek_v3` | GraphTrainer DeepSeek V3 | `torchtitan/experiments/graph_trainer/README.md` |
| `torchtitan_recipes.rl.alphabet_sort` | RL 字母排序 | `torchtitan/rl/examples/alphabet_sort/` |
| `torchtitan_recipes.rl.dapo_math` | RL DAPO 数学 | `torchtitan/rl/examples/dapo_math/` |
| `torchtitan_recipes.rl.search_r1` | RL Search-R1 | `torchtitan/rl/examples/search_r1/` |
| `torchtitan_recipes.rl.verifiers_dapo_math` | RL verifiers DAPO 数学 | `torchtitan/rl/examples/verifiers/dapo_math/` |
| `torchtitan_recipes.rl.verifiers_terminal_bench` | RL verifiers 终端基准 | `torchtitan/rl/examples/verifiers/terminal_bench/` |
| `torchtitan_recipes.sft.slake` | SLAKE 医学 VQA LoRA SFT（Muse Spark、Muse Glimmer） | `torchtitan_recipes/sft/slake.py` |
| `torchtitan_recipes.decision.qwen3` | 类型化决策（Jev 风格）微调 | `torchtitan_recipes/decision/README.md` |
| `torchtitan_recipes.torchft.llama3` | TorchFT 复制训练 | `torchtitan/experiments/torchft/README.md` |

通过精确的导入路径选择配方：

```bash
MODULE=torchtitan_recipes.models.llama3 CONFIG=llama3_8b ./run_train.sh
```

内置的可选组件覆盖位于 `torchtitan_recipes.overrides` 下。通用覆盖机制记录在 `torchtitan/config/OVERRIDE.md` 中，也接受外部模块。