为加速围绕 `torchtitan` 的贡献与创新，我们新增了这个实验性文件夹。以下是通用贡献指南，期待你的贡献！

## 贡献指南

我们提供 `experiments/` 文件夹来托管对 `torchtitan` 有显著价值的实验，遵循以下原则。我们将 `experiments` 之外的 `torchtitan` 部分称为 `core`。
1. `experiments` 中的每个子文件夹是一个实验，主题明确且可灵活设定，例如
    - 一个新模型，或最好是新的模型架构，及其训练基础设施（包括并行化功能）。贡献新模型的说明见[此处](/torchtitan/models/README.md)。
    - 对 `torchtitan` 现有基础设施的增强或补充。
2. 贡献者有责任论证实验的价值。`torchtitan` 团队将逐案评审提案。作为贡献的一部分，贡献者应提供文档，清晰展示实验的动机与创新点，包括性能与损失收敛报告。
3. 实验应尽可能复用现有 `torchtitan` 代码，如 [`components/`](../components/) 中的模块、[`models/base/`](../models/base/) 中的模型基类，以及 [`train.py`](../train.py)。主要扩展点为[新模型](../models/README.md)、[组件覆盖](../config/OVERRIDE.md)和[自定义 trainer 配置](../config/README.md#adding-configuration-fields)。
    - 扩展点可能变更。若复用任何组件时遇到问题，请贡献者提供反馈，而不是简单复制粘贴。
    - 现有组件的复用程度以及重复代码是否合理，也是实验能否被接受的标准之一。
4. 每个实验独立于其他实验，可以有自己的依赖（在[核心依赖](../../requirements.txt)之上）和自己的测试。实验不应包含厂商特定代码，例如用专有语言编写的 kernel。这些可以作为依赖在外部托管。
5. `experiments` 对 `core` 的依赖是单向的。`experiments` 中的任何内容对 `core` 成功运行都是可选的。特别是，`core` 的开发不会被 `experiments` 中的故障阻塞。我们将利用 GitHub 的 [CI 机制](https://docs.github.com/en/actions/writing-workflows/workflow-syntax-for-github-actions#onpushpull_requestpull_request_targetpathspaths-ignore)定期测试实验，且仅在实验本身受 PR 影响时触发。
6. 每个实验需要有一个负责人。负责人负责与 `torchtitan` 团队合作，维护实验的质量与健康度，包括
    - 使实验适配 `core` 的变更并修复失败的测试，最迟在下一个官方 `torchtitan` 发布版本前完成；
    - 及时响应 GitHub Issue 和问题。
7. `torchtitan` 团队保留移除实验的权利。特别是，实验应在以下情况被移除
    - 已完成其目的（例如提供发现，或将某些功能上游到 `core` 或 PyTorch 等），或
    - 已过时（例如不再维护）。


## 当前实验

| 实验 | 测试状态 | 负责人 |
| ----- | ----: | ----: |
| [graph_trainer](./graph_trainer/) | [![GraphTrainer 8 GPU Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer.yaml?query=branch%3Amain) [![GraphTrainer H100 8 GPU Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer_h100.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_graph_trainer_h100.yaml?query=branch%3Amain) | [@SherlockNoMad](https://github.com/SherlockNoMad) |
| [torchft](./torchft/) | [![TorchFT 8 GPU Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_torchft.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_torchft.yaml?query=branch%3Amain) | [@d4l3k](https://github.com/d4l3k) [@tushar00jain](https://github.com/tushar00jain) |
| [transformers_modeling_backend](./transformers_modeling_backend/) | [![Transformers modeling backend 8 GPU Integration Tests](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_transformers_modeling_backend.yaml/badge.svg?branch=main)](https://github.com/pytorch/torchtitan/actions/workflows/integration_test_8gpu_transformers_modeling_backend.yaml?query=branch%3Amain) | [@HosseinKaviani-H](https://github.com/HosseinKaviani-H) |