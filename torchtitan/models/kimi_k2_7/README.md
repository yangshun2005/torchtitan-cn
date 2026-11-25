# Kimi K2.7

Kimi K2.5、K2.6 和 K2.7-Code 共享同一架构：**DeepSeek-V3 风格**解码器（多头潜在注意力 + 混合专家）搭配 **MoonViT3d** 视觉编码器。

## 前置条件

安装额外依赖：

```bash
pip install av torchvision
```

## 架构

- **解码器** — DeepSeek-V3（MLA + MoE）。
- **视觉编码器** — MoonViT3d：线性 patch 嵌入、可学习 2D 空间位置嵌入（视频额外加正弦时间项）、2D RoPE、pre-norm transformer 块、时间均值池化 + 2x2 空间合并，最后经 2 层 MLP 投影器映射到解码器隐藏维度。
- **多模态前向** — 投影后的视觉嵌入按共享媒体占位符 token 的连续段散布到文本嵌入序列中。

`KimiK2Model` 是纯文本语言模型（供纯文本 `moonlight-16B-A3B` 变体使用）；`KimiK25Model` 额外包含视觉编码器。

## 模型变体

| 变体 | LLM 维度 | 层数 | 头数 | 专家数（top-k） | ViT 维度 | ViT 层数 | ViT 头数 |
|---------|---------|--------|-------|-----------------|---------|------------|-----------|
| debugmodel | 256 | 6 | 16 | 8 (top-3) | 256 | 4 | 4 |
| moonlight-16B-A3B | 2048 | 27 | 16 | 64 (top-6) | — | — | — |
| Kimi-VL-A3B | 2048 | 27 | 16 | 64 (top-6) | 1152 | 27 | 16 |
| Kimi-K2.5 | 7168 | 61 | 64 | 384 (top-8) | 1152 | 27 | 16 |

原 recipe 使用的对应资源目录为 `assets/hf/Moonlight-16B-A3B`、`assets/hf/Kimi-VL-A3B` 和 `assets/hf/Kimi-K2.5`。它们对用户自建 recipe 仍有参考价值。

## QK 裁剪

所有 recipe 均使用现有 DistMuon 参数组，并采用 Kimi K2 技术报告中的 QK 裁剪更新。它们记录每个 MLA 头的最大注意力分数，在数据并行和上下文并行 rank 间归约这些最大值，并在每次优化器步进后裁剪 Q/K 投影权重。该更新目前使用报告中的值 `threshold=100.0` 和 `alpha=0.5`。

## 支持的并行方式

| 特性 | 说明 |
|---------|-------|
| FSDP / HSDP | 解码器按层分片。无 PP 时，视觉编码器是独立的 FSDP 单元；有 PP 时，它属于第一阶段的根 FSDP 单元 |
| 张量并行（TP） | 模型已支持，但 DistMuon recipe 目前拒绝 TP 产生的 `_StridedShard` 布局（[#3353](https://github.com/pytorch/torchtitan/issues/3353)） |
| 专家并行（EP） | DeepSeek-V3 路由专家 + 共享专家。当 `edp_shard_size * ep_size > num_experts` 时，DistMuon 保留 `ep` 轴的 `Shard(0)` 存储布局，同时将 `edp_shard` 轴的存储布局从 `Shard(1)` 重分布为 EP 之后计算的 `Shard(0)`（[#4122](https://github.com/pytorch/torchtitan/pull/4122)） |
| 流水线并行（PP） | 视觉编码器并入第一阶段；支持 1F1B 和 Interleaved1F1B 调度。DistMuon 还要求每个阶段至少拥有一层 transformer：仅含 `norm` 和 `lm_head` 的阶段没有 Muon 矩阵，其参数组为空，优化器构建会失败并报 `matched no parameters`。这需要阶段数接近层数（Kimi K2 为 61），因此在实际深度下不会出现 |

## 数值检查

HuggingFace 对比覆盖 Kimi-VL 兼容变体，不覆盖 1T 的 K2.x 检查点。对于完整的文本+图像 float32 执行，测得：

- 视觉特征余弦相似度：`0.999977`
- 正常端到端末 token logits：KL `4.3e-2`，top-1 匹配，top-5 4/5
- 将专家路由固定为 HuggingFace 的选择：KL `5.3e-4`，top-1 匹配，top-5 5/5

路由固定结果是隔离非路由数学的诊断手段，不是正常的端到端一致性结果。

- **并行正确性**：无并行 / FSDP / FSDP+EP 的 logits 逐位一致（最大差异 `0.0`）；FSDP+EP+TP（含 SP）在 bf16 容差内。

测试脚本：
- `scripts/checkpoint_conversion/numerical_tests_kimi.py`

## TODO

- 添加视频数据集训练流水线。
- 添加 INT4（compressed-tensors）检查点加载。已发布的 K2.5、K2.6 和 K2.7-Code 1T 检查点为 INT4 分组量化；继承的 DeepSeek-V3 适配器仅处理 FP8 块缩放，因此 1T 配置可从零训练，但尚无法加载这些检查点。
- 添加上下文并行（CP）支持。