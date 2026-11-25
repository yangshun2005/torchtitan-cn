# Muse Glimmer

这是 Muse Glimmer 的 Torchtitan 实现。
它同时移植了**文本**解码器（`MuseGlimmerModel`）和
**视觉/多模态**模型（`MuseGlimmerMultimodalModel`，后者持有视觉
编码器 + 适配器，并在 `forward` 中运行它们）。

## 快速开始

```bash
# 8 GPUs
NGPU=8 MODULE="torchtitan_recipes.tests.models.muse_glimmer" CONFIG="muse_glimmer_debugmodel" ./run_train.sh
```

调试配置使用 `c4_test` 数据集，因此请确保测试分词器
资源存在（`./tests/assets/tokenizer`）。

## 模型规格

已验证的文本预训练配方位于
[`torchtitan_recipes/models/muse_glimmer.py`](../../../torchtitan_recipes/models/muse_glimmer.py)。
调试配置位于 `torchtitan_recipes.tests` 下；模型规格在
[`__init__.py`](./__init__.py) 中构建。

| 规格 | dim | 层数 | 头数（Q/KV） | head_dim | 词表 | 视觉 |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `debugmodel` | 256 | 8 | 4 / 2 | 64 | 2048 | 否 |
| `30B` | 6656 | 52 | 32 / 2 | 128 | 202048 | 否 |
| `debugmodel_mm` | 256 | 8 | 4 / 2 | 64 | 2048 | 是 |
| `30B_mm` | 6656 | 52 | 32 / 2 | 128 | 202048 | 是 |

## 架构要点

- **以增益为中心的 RMSNorm**（`RMSGainCenterNorm`）：有效缩放为
  `weight + gain_center`，`weight` 初始化为 0。
- **无缩放 RMSNorm**，用于 QK-norm 和 token 嵌入归一化。
- **GQA 注意力**：RoPE 之前做 QK-norm，使用调优的 query 缩放，并在 `wo` 之前
  应用 sigmoid 输出门。
- **iRoPE**：NoPE 层跳过 RoPE（`every_n_layers_nope = 4`），使用复数
  RoPE 后端，`theta = 500_000`。
- **逐层滑动窗口与全局注意力**按循环模式切换；
  目前通过 **Flex** 注意力后端实现。
- **后归一化残差**和 SwiGLU FFN。
- **`SoftCappedLinear` lm_head**：在 head 内部应用输出乘数 + tanh 软上限
  （因此在 `ChunkedLossWrapper` 下仍保持正确）。

## 多模态（视觉）

`*_mm` 规格构建 `MuseGlimmerMultimodalModel`，它持有视觉编码器 + 适配器
（[共享 Spark 实现](../muse_spark/vision_encoder.py)），并在 `forward` 中运行它们，
构建打包的视觉特征，并使用在 token 分片之前准备好的
bank 索引将其收集到 token 嵌入中。多模态融合采用 TP 复制；
第一个解码器层恢复标准解码器布局（启用时为序列并行）。

## 并行支持

并行由 [`model.py`](./model.py) 通过共享的模型
生命周期（FSDP、HSDP、TP、SP、CP、`torch.compile` 和 PP）应用。Muse
Glimmer 专属模块的分片定义在 [`sharding.py`](./sharding.py) 中。多模态路径支持 CP、
TP+CP+SP 和 TP+CP+PP+SP，同时保持视觉计算在
CP 上复制。