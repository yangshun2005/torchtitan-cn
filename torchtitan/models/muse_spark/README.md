# Muse Spark Open

本目录包含 Muse Spark Open 的 TorchTitan 原生文本与模型级多模态实现。

已发布的检查点为
[`meta-models/Muse-Spark-Open-816B-A42B`](https://huggingface.co/meta-models/Muse-Spark-Open-816B-A42B)。
将资产暂存到本地 `./assets/hf/Muse-Spark-Open-816B-A42B`，供生产与 SFT 配方使用。

## 快速开始

```bash
NGPU=4 MODULE=torchtitan_recipes.tests.models.muse_spark CONFIG=muse_spark_debugmodel ./run_train.sh
```

运行 10 步调试，混合本地 CC12M 测试图像与合成音频、视频数据集，Grain 权重相等：

```bash
NGPU=8 MODULE=torchtitan_recipes.tests.models.muse_spark CONFIG=muse_spark_debugmodel_mm ./run_train.sh
```

该配方将混合样本打包为 512 token 的微批次，并使用 DP2/CP2/TP2/EP2 与序列并行。

## 模型规格

| 规格 | 模态 | dim | 层数 | 头数（Q/KV） | 默认上下文 | 专家数 / top-k |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `debugmodel` | 文本 | 256 | 8 | 8 / 4 | 512 | 8 / 2 |
| `debugmodel_mm` | 文本、视觉、音频 | 256 | 8 | 8 / 4 | 512 | 8 / 2 |
| `816B-A42B` | 文本 | 8192 | 62 | 128 / 16 | 32768 | 256 / 8 |
| `816B-A42B_mm` | 文本、视觉、音频 | 8192 | 62 | 128 / 16 | 32768 | 256 / 8 |

生产规格默认使用 32,768 token 的训练序列，并接受显式序列长度，最高可达已发布检查点的上限 1,048,576。

## 模态与检查点

文本规格构建 `MuseSparkModel`，即不含编码器的解码器；HF 加载从多模态检查点读取解码器，忽略编码器权重，因此从文本规格保存的检查点不含编码器权重。`_mm` 规格构建 `MuseSparkMultimodalModel`，它拥有视觉编码器和音频编码器。将其 `model.dataset_modalities` 设为当前数据集在各 DP rank 上能提供的模态，且每个 rank 取值相同：

- `("vision",)`：图像和视频，使用一次共享的视觉前向。
- `("audio",)`：音频。
- `("vision", "audio")`：两者；多模态规格的默认值。

纯文本训练使用文本规格；`dataset_modalities` 为空会报错。

CC12M 激活视觉；调试 Grain 混合激活两者。列出的模态需要其编码器，未声明的媒体输入会抛出 `ValueError`。

列出的编码器在每个 DP rank 上运行，媒体缺失时使用虚拟输入，包括所有 rank 都无媒体的步骤。这种静态选择避免了媒体存在性集合通信，保持 FSDP all-gather 与 reduce-scatter 对齐；依赖 rank 的跳过可能导致挂起。虚拟输出为 token 嵌入添加零值依赖，使反向集合通信得以运行，同时不改变文本激活。

未列出的编码器被跳过，但仍保留在检查点中。整个训练阶段未使用的编码器其参数梯度保持为 `None`，因此 AdamW 不改变其权重和优化器状态。保持各阶段的编码器配置与优化器参数分配一致。若自定义多模态配置省略某个编码器，仅模型的 native/HF 加载会加载其余模块的权重。后续检查点将排除被省略的编码器。

## 架构要点

- GQA，带逐头 Q/K RMSNorm。第 2、6、10、...、62 层（从 1 计数）使用无 RoPE 的全局注意力。其余层使用带复数 RoPE 的滑动窗口注意力。注意力支持 FlexAttention 和 varlen 注意力，输出投影前有输出 RMSNorm 和学习到的 sigmoid 输出门。
- 增益中心 RMSNorm 和学习到的保范残差门。注意力与专家计算使用 BF16 输入和权重。路由分数及穿过解码器块的残差流使用 FP32。
- 压缩 MoE，含 256 个 SwiGLU 专家。路由器以 8192 特征读取输入。专家计算时，输入投影到 4096 特征并归一化。Token 按所选专家分组。每个专家输出归一化，然后加权求和。最终投影恢复 8192 特征。
- 路由器计算 FP32 sigmoid 分数，使用分数 + 专家偏置为每个 token 选择 8 个专家。组合权重使用所选分数（不含偏置），在 8 个所选专家间归一化。生产缩放因子为 1.0。专家偏置根据 token 计数更新，以平衡专家使用，无需辅助损失。每次更新去除均值，系数为 `1e-3`。
- RMS 归一化嵌入，以及缩放、tanh 软截断的语言模型头。

生产和调试规格均在每一层使用 MoE。

## 并行支持

| 特性 | 状态 |
| --- | --- |
| 单设备 | 支持 |
| FSDP | 可单独使用，或与 EP/TP+EP 一起使用；活跃编码器使用虚拟输入，以保持各 DP rank 参与对齐 |
| 专家并行 | 支持标准 all-to-all 调度器、DeepEP、HybridEP 或 Dist-MoE |
| 张量并行 | 支持 `EP >= TP`，包括不等度数 |
| 异步张量并行 | 支持 BF16 投影 |
| 序列并行 | 在 TP 和 EP 均启用时支持 |
| 流水线并行 | 支持 |
| 上下文并行 | FlexAttention 配合 Ulysses 或 K/V all-gather；varlen 配合 Ulysses |
| CP 负载均衡 | 头尾和 PTRR 配合 K/V all-gather FlexAttention；Ulysses 无需负载均衡器 |
| 激活检查点 | 通过 `torch_remat` 支持 full 和 region 策略 |
| 损失编译 | 支持 |
| 模型编译 | 区域编译 |
| CUDA graphs | 仅文本配合 Dist-MoE 或 HybridEP；多模态支持是 TorchTitan 的缺口；标准 all-to-all EP 和 DeepEP 训练需禁用 graphs |
| 无 EP 的 TP | 不支持 |

生产 TP 必须整除 16 个 KV 头。EP 必须至少为 TP，并整除 256 个专家和 `dp_shard * CP * TP` 个 rank。

Ulysses CP 要求 `TP * CP` 整除 16 个 KV 头，且 `context_parallel_load_balancer=None`。

K/V all-gather CP 要求每微批次 token 数可被 CP 度数乘以 FlexAttention 查询块大小整除。媒体输入在 CP 间保持复制；token 对齐的 bank 索引选择各 rank 的本地嵌入。

Dist-MoE 需要 Blackwell GPU 和可选的 `dist_moe` 包。Dist-MoE 负责调度、专家计算和合并。

在 Hopper 上（Dist-MoE 不可用），HybridEP 是支持 CUDA graphs 的 EP 路径。容量因子 `1.0` 时，其内存约为标准 all-to-all 调度器的 2 倍。在 8x H100 的放大调试运行中，无 graphs 时比 all-to-all 快约 15%，有 graphs 时快约 60%。`torchtitan_recipes/tests/models/muse_spark.py` 中的 `muse_spark_debugmodel_hybridep` 展示了配合 `TokenDispatcherTransform` 和 CUDA graphs 的设置。其非阻塞容量因子 `1.0` 预留最坏情况，因此不丢弃 token，代价是内存。选择更低因子见 [`models/common/moe/README.md`](../common/moe/README.md)。HybridEP 已验证配合 FSDP、TP+EP 和 Ulysses CP+EP。HybridEP 的合并会在加权专家输出前将路由分数舍入到 BF16。

DeepEP 在训练中使用其紧凑、主机同步的路径，因此必须禁用 CUDA graphs。其合并内核要求 MoE 压缩维度为 256 的倍数：生产规格使用 4096，但 `debugmodel` 使用 128，无法配合 DeepEP 运行。DeepEP 已验证配合 FSDP、TP+EP 和 Ulysses CP+EP。

## TODO

- 在配置阶段于 `DeepEPTokenDispatcher.Config` 中拒绝非 256 倍数的 DeepEP 隐藏大小，而非在首次前向时。
- 添加非合成音频和视频训练配方。
- 在更长上下文和生产规模下基准测试 CP 负载均衡和 PP/CP 通信。添加跨 PP 阶段布局的 native 和 Hugging Face 检查点保存/恢复测试。
- 使用生产波形长度和模态混合基准测试音频块 MLP 的张量并行。初始保持卷积、归一化和残差流 TP 不变，使用列并行 `linear1` 和行并行 `linear2`；验证分片扩展激活的收益超过每块归约开销。