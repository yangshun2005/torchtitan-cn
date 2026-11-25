# 如何在 `torchtitan` 中使用检查点

你可能希望在 `torchtitan` 中启用检查点，以便在训练期间获得更好的容错能力，或更方便地在 `torchtitan` 与其他库之间导入和导出权重。`torchtitan` 对其他检查点格式提供不同程度的支持，详见下文。

## 训练期间使用检查点的通用指南

1. 启用检查点
在 recipe 函数中配置检查点设置：
```python
checkpointer=CheckpointManager.Config(
    interval=500,
),
```
检查点在 recipe 中配置，而非通过 CLI。

2. 仅保存模型
将 `last_save_model_only` 设为 `True` 后，检查点只包含模型，不包含优化器状态和额外训练状态，因此体积更小。
```python
checkpointer=CheckpointManager.Config(
    interval=500,
    last_save_model_only=True,
),
```
这种最终保存无法恢复训练，因此会作为导出写入 `exports/` 下，而不是放在可恢复的检查点旁边。自动恢复只识别 `step-N` 目录，所以用更大的 `training.steps` 重新运行时，会从最新的完整检查点继续：
```
checkpoint/
|-- step-500/          # 完整检查点，可恢复
|-- step-1000/         # 完整检查点，可恢复
`-- exports/
    `-- step-1200/     # 最终仅模型保存（training.steps=1200）
```
要加载导出，将 `initial_load_path` 指向它（例如 `.../checkpoint/exports/step-1200`）；`initial_load_model_only` 默认为 `True`。请从 `checkpointer.folder` 中不含 `step-N` 的任务运行；否则会加载最新的 `step-N`，并忽略 `initial_load_path`。种子检查点保持在 `step-0`。

3. 选择导出精度
模型状态默认为 `float32`。你可以选择以更低精度格式导出检查点，例如 `bfloat16`。
```python
checkpointer=CheckpointManager.Config(
    interval=500,
    last_save_model_only=True,
    export_dtype="bfloat16",
),
```

4. 从检查点加载时排除特定键
某些情况下，你可能希望从先前训练的检查点部分加载，并修改某些设置，例如 GPU 数量或当前步数。为此，可使用 `exclude_from_loading` 参数指定哪些键不加载。
```python
checkpointer=CheckpointManager.Config(
    exclude_from_loading=["dataloader", "lr_scheduler"],
),
```

权重 EMA（`optim.ema`）为 `decays`（固定衰减）和 `half_life_fractions`（半衰期调度）的每个条目保留一份副本。每份副本以 `ema.state.<fqn>.ema_params.<ema_key>` 的形式保存在 `"ema"` 键下，其中键由值派生，例如 `decay_0p999` 或 `half_life_0p05`。
```python
optim=Optim.Config(
    ema=EMA.Config(decays=[0.999, 0.9999], half_life_fractions=[0.05]),
),
```

在运行中途开启权重 EMA，或向 `decays`/`half_life_fractions` 添加条目，需要同样的应急手段：现有检查点没有匹配的 EMA 键，因此加载会失败，直到你排除一次 `"ema"`，之后每份 EMA 副本都会从加载的权重冷启动。
```python
checkpointer=CheckpointManager.Config(
    exclude_from_loading=["ema"],   # 仅在启用 EMA 后的首次恢复时使用
),
```
之后请移除该项。若保留，它会在之后*每次*恢复时冷启动 EMA，每次都丢弃全部 EMA 历史（会记录为警告）。

5. 检查点配置示例
```python
checkpointer=CheckpointManager.Config(
    interval=10,
    load_step=5,
    last_save_model_only=True,
    export_dtype="bfloat16",
),
```

更详尽且最新的检查点配置选项列表见 `torchtitan/components/checkpointer/base.py`（`BaseCheckpointManager.Config`）。其中包括 `async_mode`：DCP 管理器支持全部三种模式，`TorchCheckpointingManager` 支持 `disabled` 和 `async_with_pinned_mem`。

## 使用 `TorchCheckpointingManager` 的远程存储

`TorchCheckpointingManager` 可通过 fsspec 将检查点保存在远程存储上。
将 `folder` 指向 URI 并安装其驱动，例如 `gs://` 需要 `gcsfs`：

```python
checkpointer=TorchCheckpointingManager.Config(
    folder="gs://my-bucket/run-1/checkpoint",
    interval=500,
),
```

仅验证了 `gs://`。带主机或端口的 URI（如 `hdfs://host:8020/path`）不受支持：主机部分会被当作路径的一部分读取。

定期保存、最终保存、恢复、`keep_latest_k` 保留、Hugging Face 导出（`last_save_in_hf`）和 Hugging Face 加载（`initial_load_in_hf`）都通过远程存储进行。`initial_load_path` 可以位于 `folder` 之外的存储上，本地 `hf_assets_path` 也可与远程 `folder` 配合使用，例如 HF 权重在本地磁盘、检查点在 `gs://`。

将 `hf_assets_path` 保留在本地磁盘，且即使权重在远程也要包含 safetensors 索引：
```bash
python scripts/download_hf_assets.py --repo_id <repo_id> --assets tokenizer index
```
状态字典适配器从 `hf_assets_path` 读取 `model.safetensors.index.json`，而非从 `initial_load_path`。

在 GCS 上，请使用具有分层命名空间（HNS）的存储桶。在扁平存储桶上，检查点通过逐个复制对象来提交，因此提交中途被抢占的任务可能留下一个 `step-N`，恢复时会选中它却加载失败。

## 创建种子检查点
有时需要创建种子检查点，以便从第 0 步初始化模型。
例如，多设备上的元初始化很难（甚至不可能）复现单设备上的初始化。
种子检查点在单个 CPU 上完成模型初始化，并可通过 DCP 重分片从另一任务在任意数量的 GPU 上加载。

要创建种子检查点，定义一个 recipe，设置
`create_seed_checkpoint=True`、非 `None` 的 `checkpointer`，并将所有并行度设为 1。然后在单个设备上运行该配置。

## 转换支持

### HuggingFace
`torchtitan` 提供两种使用 Hugging Face 模型的方式：在训练期间直接保存和加载 Hugging Face 检查点，或使用示例转换脚本在 CPU 上直接重排模型权重格式。

1. 你可以在训练期间通过 recipe 中的 `checkpointer.last_save_in_hf` 和 `checkpointer.last_save_model_only` 直接保存 Hugging Face 模型权重。最终保存写入 `<checkpointer.folder>/exports/step-<N>`。要从 Hugging Face safetensors 文件直接加载 `torchtitan` 训练会话，设置 `checkpointer.initial_load_in_hf`，并将 `hf_assets_path` 或 `checkpointer.initial_load_path` 设为包含 Hugging Face 检查点的目录。若两者都设置，`checkpointer.initial_load_path` 优先于 `hf_assets_path`。若 `checkpointer.folder` 已包含有效检查点，训练会从该文件夹恢复并忽略 `initial_load_in_hf` / `initial_load_path`（容错重启）。首次运行（空文件夹）使用初始加载。

2. 若无需运行训练循环直接重排权重格式，运行相应的转换脚本。命名以 `torchtitan` 为中心，例如 convert_from_hf 表示 hf->tt。`convert_ema_to_hf` 从同一检查点导出 EMA 权重而非训练权重；它接受相同参数，外加 `--ema_key` 选择一份 EMA 副本（例如 `--ema_key half_life_0p05`；检查点只有一份副本时可选），若检查点无 EMA 状态则报错。注意 `last_save_model_only`（默认）仅在最后一步写入模型权重，因此请从定期检查点导出 EMA，或将其设为 `False`。

```bash
python ./scripts/checkpoint_conversion/convert_from_hf.py <input_dir> <output_dir> --model_name <model_name> --model_flavor <model_flavor>
python ./scripts/checkpoint_conversion/convert_to_hf.py <input_dir> <output_dir> --hf_assets_path ./assets/hf/Llama-3.1-8B --model_name <model_name> --model_flavor <model_flavor>
python ./scripts/checkpoint_conversion/convert_ema_to_hf.py <input_dir> <output_dir> --hf_assets_path ./assets/hf/Llama-3.1-8B --model_name <model_name> --model_flavor <model_flavor> --ema_key <ema_key>
# 例如
python ./scripts/checkpoint_conversion/convert_from_hf.py ~/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/8cde5ca8380496c9a6cc7ef3a8b46a0372a1d920/ ./initial_load_path/ --model_name llama3 --model_flavor 8B
```

### Torch

本指南将引导你完成将 `torchtitan` 检查点转换为可加载到 pt 格式的步骤。

1. 检查点配置
```python
checkpointer=CheckpointManager.Config(
    interval=10,
    last_save_model_only=True,
    export_dtype="bfloat16",
),
```

2. 保存最终检查点\
完成上述设置后，训练结束时的最终检查点将仅包含模型，并使用所需的导出 dtype。但如果尚未到达最后一步，仍会保存完整检查点以便恢复训练。

3. 将分片检查点转换为单文件\
最后，获得最后一个检查点后，可使用以下命令将分片检查点转换为单个 .pt 文件。

```bash
python -m torch.distributed.checkpoint.format_utils dcp_to_torch torchtitan/outputs/checkpoint/exports/step-1000 checkpoint.pt
```


就是这样。你现在已成功将分片的 `torchtitan` 检查点转换为可用于 pytorch 格式。