# Kimi K3.1

Kimi K3.1 Pro 保留 Kimi K3 主干，在相同层位置将 24 个 Gated MLA 层替换为 Gated DeepSeek Sparse Attention（DSA）。本文件夹仅存放 K3.1 各变体；模型和 DSA 层（`torchtitan/models/kimi_k3/dsa.py`）位于 `torchtitan/models/kimi_k3/`。

## 前置条件

与 [Kimi K3](../kimi_k3/README.md) 相同。

## 架构

每个 DSA 层保留 Kimi K3 的 MLA 投影和输出门，并新增一个索引器：32 个头、维度 128，每个查询选取 2048 个 token。Top-k 选择和索引器损失来自 `torchtitan/models/deepseek_v4/compressor.py`。索引器仅由索引器损失训练。`build_model_config(..., indexer_loss_coeff=...)` 设置其系数（默认 0.01）；设为 0 则关闭该损失。

稀疏注意力使用 Attention Gym 的 `gather_attn`，输入为 `(latent, rope)` 对。

## 支持的并行方式

| 特性 | 说明 |
|---------|-------|
| FSDP2 | 支持 |
| 张量 / 序列并行 | 支持；Attention Gym 配对内核要求每个 rank 有 16 到 128 个查询头 |
| 专家并行 | 支持 |
| 流水线并行 | 支持 |
| 上下文并行 | 不支持 |

## 数值一致性

`KimiK3StateDictAdapter` 映射已发布的 Hugging Face 检查点，包括索引器权重。

K3.1 未发布 Hugging Face 模型代码，因此前向参考为 vLLM 的 K3.1 DSA 层。一个调试模型 DSA 层（随机权重，1024 个 token）与其一致：fp32（参考内核）下 top-k 相同，相对 L2 误差为 6.6e-7；bf16 下 top-k 重叠率为 99.6%，在 top-k 相同的情况下相对 L2 误差为 7.2e-3。