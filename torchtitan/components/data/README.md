# 数据集

TorchTitan 使用一套基于 Grain 的数据流水线，统一处理文本预训练、SFT 和图像训练。心智模型：

```text
1. 定义数据源（例如 jsonl）：
    class: SourceConfig
    output: RandomAccessDataSource | IterDataset

2. 定义数据集（对数据源应用过滤/处理）：
    class: SingleDatasetConfig
    does:   pre-filter -> process -> post-filter
    output: MapDataset | IterDataset

3. 组合数据集（可选）：
    class: 例如 FirstFitPackingConfig(dataset=DatasetMixConfig(...))
    input:  一个或多个子 DatasetConfig 值
    does:   混合、拼接和/或打包
    output: MapDataset | IterDataset

4. Dataloader：
    runtime: GrainDataLoader
    config:  GrainDataLoader.Config
    input:  MapDataset | IterDataset
    does:   按需转换为可迭代 -> 批处理 -> 整理 -> 预取
    output: TrainingMicrobatch

5. Trainer：
    input: TrainingMicrobatch
    does:  模型前向和反向传播
```

# 文本预训练

## 本地 JSONL

每个非空行必须包含一个 JSON 对象：

```json
{"title": "First", "body": "The first document."}
{"title": "Second", "body": "The second document."}
```

```python
from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    IndexedJsonlSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.text_datasets import TextProcessor


def article_text(row):
    return row["title"] + "\n\n" + row["body"]


books_ds = SingleDatasetConfig(
    source=IndexedJsonlSource.Config(
        patterns=(
            "/datasets/books/*.jsonl",
        ),
    ),
    processor=TextProcessor.Config(text_fn=article_text),
    post_filters=(lambda sample: sample is not None,),
)

books_packed_ds = ConcatThenSplitPackingConfig(dataset=books_ds)

config.dataloader = GrainDataLoader.Config(
    dataset=books_packed_ds,
)
```

## Hugging Face 数据源

对已物化的数据集使用随机访问：

```python
from torchtitan.components.data import HuggingFaceRandomAccessSource

source = HuggingFaceRandomAccessSource.Config(
    path="openai/gsm8k",
    name="main",
    split="train",
)
```

当语料不应物化时使用流式：

```python
from torchtitan.components.data import HuggingFaceStreamingSource

source = HuggingFaceStreamingSource.Config(
    path="allenai/c4",
    name="en",
    split="train",
)
```

### 多节点作业的 Hub 速率限制

从 Hub 流式读取 `allenai/c4` 发生在 trainer 初始化时：每个 rank 都会调用 `datasets.load_dataset`。大型作业（例如 48 个节点）可能触发 Hugging Face HTTP 429。

先将数据集下载到共享存储一次，然后将 `path` 指向该目录。`HuggingFaceStreamingSource.Config.path` 和 `HuggingFaceRandomAccessSource.Config.path` 接受 Hub id 或 `datasets.load_dataset` 接受的本地目录。

配方使用 `en` 配置。下载该子集（加上声明配置的 `README.md`），而不是完整的多语言转储。

```bash
huggingface-cli download allenai/c4 --repo-type dataset --include "en/*" --include "README.md" --local-dir /datasets/c4
```

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="allenai/c4",
    repo_type="dataset",
    allow_patterns=["en/*", "README.md"],
    local_dir="/datasets/c4",
)
```

`llama3_8b` 等配方使用 `ConcatThenSplitPackingConfig(dataset=DATASETS["c4"])`。替换源路径，或直接构造源：

```python
from dataclasses import replace

from torchtitan.components.data import HuggingFaceStreamingSource
from torchtitan.components.data.text_datasets import DATASETS

c4 = DATASETS["c4"]
c4 = replace(c4, source=replace(c4.source, path="/datasets/c4"))

source = HuggingFaceStreamingSource.Config(
    path="/datasets/c4",
    name="en",
    split="train",
)
```

如果语料已完全物化，使用 `HuggingFaceRandomAccessSource`，传入相同的 `path`、`name` 和 `split`。

可选地导出 `HF_TOKEN` 或 `HUGGING_FACE_HUB_TOKEN` 以获得认证的 Hub 配额。令牌不能免除在大型集群上预下载的需要。

调试配方已使用 `DATASETS["c4_test"]`（`tests/assets/c4_test/` 下的本地 JSON），不会访问 Hub。

## 添加自定义数据源——示例：预分词数据

此示例使用 token 偏移量，为每个索引暴露一个预分词文档，然后复用正常的处理和打包配置。

```python
from dataclasses import dataclass

import numpy as np

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    DatasetBuildContext,
    DatasetIterationPolicy,
    RandomAccessDataSource,
    SampleProcessor,
    SingleDatasetConfig,
    TextSequence,
)
from torchtitan.config import Configurable


class PretokenizedMemmapSource(Configurable, RandomAccessDataSource):
    @dataclass(kw_only=True, slots=True)
    class Config(Configurable.Config):
        tokens_path: str
        document_offsets_path: str

    def __init__(self, config: Config):
        self.tokens = np.memmap(config.tokens_path, dtype=np.uint32, mode="r")
        self.offsets = np.load(config.document_offsets_path)

    def __len__(self):
        return len(self.offsets) - 1

    def __getitem__(self, index):
        start, end = self.offsets[index : index + 2]
        return np.asarray(self.tokens[start:end], dtype=np.int64)


class TokensToTextSequence(SampleProcessor):
    @dataclass(kw_only=True, slots=True)
    class Config(SampleProcessor.Config):
        pass

    def __init__(self, config: Config, *, context: DatasetBuildContext):
        del config, context

    def __call__(self, token_ids, rng):
        del rng
        if len(token_ids) < 2:
            return None
        return TextSequence(
            input_ids=token_ids[:-1],
            labels=token_ids[1:],
        )


token_documents_ds = SingleDatasetConfig(
    source=PretokenizedMemmapSource.Config(
        tokens_path="tokens.bin",
        document_offsets_path="document_offsets.npy",
    ),
    processor=TokensToTextSequence.Config(),
    post_filters=(lambda sample: sample is not None,),
)

packed_tokens_ds = ConcatThenSplitPackingConfig(
    dataset=token_documents_ds,
)
```

# SFT

SFT 改变处理器和打包策略，不改变加载器：

```python
from torchtitan.components.data import (
    FirstFitPackingConfig,
    GrainDataLoader,
    HuggingFaceRandomAccessSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.text_datasets import ChatProcessor


def gsm8k_messages(row):
    return [
        {"role": "user", "content": row["question"]},
        {"role": "assistant", "content": row["answer"]},
    ]


gsm8k_ds = SingleDatasetConfig(
    source=HuggingFaceRandomAccessSource.Config(
        path="openai/gsm8k",
        name="main",
        split="train",
    ),
    processor=ChatProcessor.Config(messages_fn=gsm8k_messages),
    post_filters=(lambda sample: sample is not None,),
)

gsm8k_packed_ds = FirstFitPackingConfig(dataset=gsm8k_ds)

config.dataloader = GrainDataLoader.Config(
    dataset=gsm8k_packed_ds,
)
```

没有渲染器时，`ChatProcessor` 将分词器的聊天模板应用于单轮 `[user, assistant]` 对，创建下一 token 的输入和标签对，并将提示标签设为 `IGNORE_INDEX`。它通过使用 `add_generation_prompt=True` 渲染提示来定位提示/响应的边界，并要求该渲染是完整渲染的精确 token 前缀，否则抛出 `ValueError`。重写先前轮次的模板，或仅在上下文中合并的轮次分隔符，会破坏该假设。

对于多轮对话，显式选择模型的渲染器：

```python
from renderers import Qwen3RendererConfig

from torchtitan.components.renderer import from_renderers

processor = ChatProcessor.Config(
    messages_fn=lambda row: row["messages"],
    renderer=from_renderers(Qwen3RendererConfig()),
)
```

渲染器使用 TorchTitan 加载的分词器，一次返回带损失掩码的 token。掩码监督模型生成的 token（包括轮次终止符），排除提示 token 和模板脚手架。对话必须以 assistant 消息结束。每个对话是一个样本；打包在对话之间重置位置，而不是在轮次之间。超过 `max_context_length` 的样本会被整体丢弃。

格式化和推理保留遵循所选渲染器。例如，Qwen3 在最后一个用户查询之前省略 assistant 轮次中的推理。这些省略的 token 不获得损失。要在每个轮次的推理上训练，请在源数据集中准备单独的对话前缀。`thinking_retention` 控制渲染器的 rollout 桥接，而不是此处使用的完整渲染。

# 混合数据集

将每个权重放在其数据集旁边：

```python
from torchtitan.components.data import DatasetMixConfig, WeightedDataset

# `books_ds` 和 `code_ds` 是 SingleDatasetConfig 值。
pretraining_mix_ds = DatasetMixConfig(
    datasets=(
        WeightedDataset(dataset=books_ds, weight=0.75),
        WeightedDataset(dataset=code_ds, weight=0.25),
    ),
)
```

示例使用总和为 `1.0` 的概率，但接受任何正相对权重并在内部归一化。`weight=0.75` 与 `weight=0.25` 相邻意味着第一个数据集的抽取频率是后者的三倍。

### 当权重应计文档数时，在打包前混合文档

```python
packed_pretraining_ds = ConcatThenSplitPackingConfig(
    dataset=pretraining_mix_ds,
)
```

### 当权重应计固定长度行时，在混合前打包每个子数据集

```python
books_packed_ds = ConcatThenSplitPackingConfig(dataset=books_ds)
code_packed_ds = ConcatThenSplitPackingConfig(dataset=code_ds)

token_ratio_mix_ds = DatasetMixConfig(
    datasets=(
        WeightedDataset(dataset=books_packed_ds, weight=0.67),
        WeightedDataset(dataset=code_packed_ds, weight=0.33),
    ),
)
```

未实现基于观察到的文档或监督 token 计数的自动调整。自定义混合可以维护每个数据集的移动平均值并重新平衡其权重。

### 拼接数据集

使用拼接将有限数据集视为一个语料。每一行出现一次；数据集大小决定其比例。需要显式权重或流式数据集时改用混合。

```python
from torchtitan.components.data import DatasetConcatConfig

pretraining_corpus_ds = DatasetConcatConfig(
    datasets=(books_ds, code_ds, math_ds),
)
```

要在每个 epoch 中多次包含一个有限数据集的每一行，请在拼接中重复该子数据集：

```python
pretraining_corpus_ds = DatasetConcatConfig(
    datasets=(books_ds,) * 3 + (code_ds, math_ds),
)
```

现在每个 `books_ds` 行在组合的有限索引空间中出现三次。使用 `shuffle=True` 时，TorchTitan 在 DP 分片之前打乱该组合索引空间。

每个子数据集使用种子 `seed + index` 构建，与混合子数据集相同，因此重复副本在随机处理器中（例如增强或随机裁剪）产生独立抽取。因此，插入或重排子数据集会改变每个后续子数据集的抽取。

当期望行为是相对源选择频率而非精确的有限重复时，改用混合：

```python
pretraining_mix_ds = DatasetMixConfig(
    datasets=(
        WeightedDataset(dataset=books_ds, weight=0.6),
        WeightedDataset(dataset=code_ds, weight=0.2),
        WeightedDataset(dataset=math_ds, weight=0.2),
    ),
)
```

`GrainDataLoader.Config(repeat=True)` 重复完整的拼接或混合数据集；它不改变单个子数据集的相对贡献。

# 图像和多模态数据

图像使用相同的源、数据集、加载器、分片和检查点契约。其处理器保留模态特定的样本字典，其整理器创建模型特定的批次。

图像和视频预处理由模型固定，如同分词器：从模型的设置中设置 `Trainer.Config.vision_processor`，紧邻其特殊 token，例如 `VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS)`。其 `resize` 规则调整图像大小，`video_resize` 规则调整视频帧大小（None：模型不接受视频）。Trainer 构建一次并交给 dataloader。数据集的样本处理器用它解码、调整大小和归一化每个图像（`resize_and_normalize`，或对解码帧使用 `resize_and_normalize_video`）；整理器用它（`patchify`）将微批次的图像和视频切成补丁。

Qwen 多模态示例：

```python
from torchtitan.components.data import (
    GrainDataLoader,
    HuggingFaceStreamingSource,
    SingleDatasetConfig,
)
from torchtitan.components.data.multimodal.vision_processor import VisionProcessor
from torchtitan.components.data.multimodal.mm_collator import MultiModalCollator
from torchtitan.components.data.multimodal.mm_datasets import (
    MMSamplePackingConfig,
    VisionSampleProcessor,
    _process_cc12_wd_sample,
)
from torchtitan.models.qwen3_5 import QWEN3_5_VISION_PROCESSOR_KWARGS

mm_processor = VisionSampleProcessor.Config(
    sample_processor=_process_cc12_wd_sample,
)

mm_ds = SingleDatasetConfig(
    source=HuggingFaceStreamingSource.Config(
        path="pixparse/cc12m-wds",
        split="train",
    ),
    processor=mm_processor,
    post_filters=(lambda sample: sample is not None,),
)

packed_mm_ds = MMSamplePackingConfig(
    dataset=mm_ds,
    num_packing_bins=8,
)

config.vision_processor = VisionProcessor.Config(**QWEN3_5_VISION_PROCESSOR_KWARGS)
config.dataloader = GrainDataLoader.Config(
    dataset=packed_mm_ds,
    collator=MultiModalCollator.Config(build_mrope_positions=True),
    streaming_shuffle_buffer_size=128,
)
```

`num_packing_bins` 是保持打开的候选打包行数，不是输入样本缓冲区。

自定义图像增强应放在 `SampleProcessor` 中。

# 加载器策略

一次性配置运行范围的行为：

```python
import grain.python as grain

config.dataloader = GrainDataLoader.Config(
    dataset=packed_pretraining_ds,
    seed=42,
    shuffle=True,
    repeat=True,
    streaming_shuffle_buffer_size=1_000,
    read_options=grain.ReadOptions(
        num_threads=16,
        prefetch_buffer_size=500,
    ),
    num_prefetch_microbatches=2,
)
```

### 读取索引数据集

随机访问源变为 `MapDataset`，支持 `dataset[index]`。当后续阶段需要顺序消费样本时，它变为 `IterDataset`：

```text
所有子数据集都是 MapDataset：DatasetMixConfig 保持为 MapDataset
任一子数据集是 IterDataset：DatasetMixConfig 转换每个 MapDataset 子数据集
打包：                       按需转换其子数据集并返回 IterDataset
GrainDataLoader：            如果没有更早的阶段转换，则转换 MapDataset
```

`read_options` 控制每次 `MapDataset` 到 `IterDataset` 的转换：

```python
grain.ReadOptions(
    num_threads=16,            # 并发读取的索引样本数
    prefetch_buffer_size=500,  # 等待消费者的样本数
)
```

每次转换有自己的线程和缓冲区。全 map 混合转换一次；包含流的混合分别转换每个 map 子数据集。

### 流式打乱和就绪批次

`streaming_shuffle_buffer_size` 是为近似打乱保留的原始行数。更大的缓冲区改善混合但使用更多内存。

`num_prefetch_microbatches` 是允许等待 trainer 的完整、已整理微批次数：

```text
trainer 计算批次 10
后台线程准备批次 11 和 12
```

### 跨数据并行 rank 平衡微批次

数据并行 rank 在每个微批次上同步，因此一步以最慢 rank 的速度运行。使用文档掩码时，注意力成本随文档长度平方和增长，打包器以任意顺序发出行，因此一个 rank 的完整长度文档常与其他 rank 的短文档共享一步。`balancer` 缓冲每个 `window_size` 已整理微批次的窗口，并决定每个 rank 在每一步训练哪个微批次。`LocalBalancer` 让每个 rank 自己的窗口最重的先出。所有 rank 在同一步打开窗口，因此每一步配对成本相似的微批次，无需通信：

```python
from torchtitan.components.data import LocalBalancer

config.dataloader = GrainDataLoader.Config(
    dataset=packed_pretraining_ds,
    balancer=LocalBalancer.Config(window_size=8),
)
```

默认成本 `document_attention_cost` 是注意力段长度平方和，从 `positions` 读取，与 varlen 注意力元数据一样：文档加上填充尾部，注意力将其计算为 `max_context_length` 的段。RL 的批