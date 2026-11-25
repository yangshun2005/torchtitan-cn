# 验证与评估

`torchtitan` 为支持用户的训练目标，提供验证的直接和间接支持。直接支持由 `Validator` 类提供，它直接与训练循环交互；间接支持通过 [HuggingFace 检查点转换](../torchtitan/components/checkpointer/README.md#huggingface) 提供，供希望使用 EleutherAI 的 `lm_eval` 等外部工具进行评估的用户使用。

## 验证
对于希望在训练循环中直接执行验证的用户，我们提供 `Validator` 类，可在 recipe 函数中通过 `Validator.Config` 便捷配置。Validator 类可访问并复用训练器的许多功能，例如其并行化（包括流水线并行）。

以下是验证配置示例：

```python
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
)
from torchtitan.components.validator import Validator
from torchtitan.components.data.text_datasets import DATASETS

validator=Validator.Config(
    freq=500,
    steps=-1,
    dataloader=GrainDataLoader.Config(
        dataset=ConcatThenSplitPackingConfig(
            dataset=DATASETS["c4_validation"],
        ),
        repeat=False,
    ),
),
```

省略 `dataloader` 时默认使用此配置。

`Validator.Config.steps` 控制验证运行时长：

- `steps=-1` 将有限的验证数据集消费一次（`dataloader.repeat=False`）。各 rank 在其分片耗尽时独立停止，因此该模式要求数据并行度为 1；DP > 1 时，各 rank 可能在不同步数完成，从而在验证集合通信上挂起。若将 `steps=-1` 与 DP > 1 组合，`Validator` 会抛出异常。
- 正的 `steps` 值会在每个 rank 上运行该数量的验证步。验证器会强制 `dataloader.repeat=True`，使每个 rank 看到相同的步数。

数据并行度大于 1 时，请使用正的 `steps`。

验证遵循 `training.disable_accelerator_graphs`，并共享训练图池。
捕获顺序和张量生命周期参见 [accelerator graphs](../torchtitan/distributed/ACCELERATOR_GRAPH.md)。

## 第三方评估
通过 `./scripts/checkpoint_conversion/convert_to_hf.py`，`torchtitan` 支持将检查点从 DCP 转换为 safetensors 格式。使用该脚本，用户可借助支持 HuggingFace 的外部库（例如以 `vllm` 为后端的 `lm_eval`）在训练之外进行高效评估。

### `lm_eval` 配合 `vllm` 的用法示例：
使用此特定设置时，请确保包含 HuggingFace 的 `config.json` 文件（转换脚本或 `last_save_in_hf` 选项不提供该文件）。可通过运行 `python ./scripts/download_hf_assets.py --repo_id meta-llama/Llama-3.1-8B --assets config` 下载 HF 配置文件。

注意，pip 安装 `lm-eval` 可能破坏 `torchtitan` 开发环境，因此建议创建独立环境。
```bash
pip install "lm-eval[vllm]"
lm_eval --model vllm \
    --model_args pretrained=./outputs/checkpoint/exports/step-1000,tensor_parallel_size=8,dtype=auto,gpu_memory_utilization=0.8, \
    --tasks mmlu \
    --batch_size auto
```
|      Groups      |Version|Filter|n-shot|Metric|   |Value |   |Stderr|
|------------------|------:|------|------|------|---|-----:|---|-----:|
|mmlu              |      2|none  |      |acc   |↑  |0.6209|±  |0.0038|
| - humanities     |      2|none  |      |acc   |↑  |0.5481|±  |0.0066|
| - other          |      2|none  |      |acc   |↑  |0.7045|±  |0.0078|
| - social sciences|      2|none  |      |acc   |↑  |0.7351|±  |0.0078|
| - stem           |      2|none  |      |acc   |↑  |0.5357|±  |0.0085|