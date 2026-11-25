# 测试配方

这些配方用于在 CI 中测试 TorchTitan，并非生产训练推荐配置。

- `models/` 包含各模型可复用的调试配置。
- `graph_trainer/` 和 `torchft/` 包含子系统专属的测试配方。
- `rl/` 包含 RL 示例的调试模型与 CI 配方，每个示例一个模块（已验证的配方位于 `torchtitan_recipes/rl/`）。
- `transformers_modeling_backend.py` 包含该独立测试面的测试配方。

若配置仅改动模型本地的测试设置，就与对应模型放在一起。若配置代表跨模型或硬件测试矩阵中的一项，则放在 `tests/integration_tests/` 中对应测试定义旁边，这样新增集成测试只需改一个文件。