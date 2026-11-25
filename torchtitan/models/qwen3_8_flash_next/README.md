# Qwen3.8-Flash-Next

## 概述

Flash-Next 复用 Qwen3.5 的 Gated DeltaNet（GDN）、视觉编码器和模型生命周期。

- **门控残差：** 四条隐藏流；每个注意力/MoE 子层读取一个激活，并向全部四条流写入门控更新。
- **混合注意力：** 三个 GDN 层后接一个 Qwen 稀疏注意力（QSA）层。QSA 选择完整的四 token 块加因果尾部。
- **MoE：** 路由专家和带 sigmoid 门的共享专家。
- **PLE：** 哈希 bigram/trigram 嵌入在第 2 层注入词汇特征。
- **视觉：** 图像/视频特征在解码器之前替换占位嵌入；MRoPE 使用时间、高度和宽度坐标。

## 前置条件

多模态配方需安装额外依赖：

```bash
pip install -r .ci/docker/requirements-vlm.txt
```

## 模型变体

| 变体 | 解码器宽度 | 层数 | QSA 头数 / KV 头数 | QSA 头宽 | 专家数 / top-k |
| --- | ---: | ---: | --- | ---: | --- |
| `debugmodel` | 256 | 8 | 4 / 2 | 64 | 8 / 2 |
| `debugmodel_mm` | 256 | 8 | 4 / 2 | 64 | 8 / 2 |
| `125B-A6B` | 2,560 | 48 | 24 / 2 | 256 | 512 / 10 |

`debugmodel` 构建纯文本的 `Qwen38FlashNextModel`；`debugmodel_mm` 和 `125B-A6B` 构建 `Qwen38FlashNextMultimodalModel`，后者增加视觉编码器。

发布的视觉编码器有 27 层、宽度 1,152、16 像素 patch。其 merger 产生 2,560 宽的解码器嵌入。文本词表有 248,320 个条目。
[`qwen38_flash_next_125b_dist_moe`](../../../torchtitan_recipes/models/qwen3_8_flash_next.py) 配方在 16 块 GB300 上微调发布的解码器（每 GPU 每步 8 个 16K token 微批），使用 Dist-MoE 和加速器图，自然路由下每 GPU 379 TFLOP/s；专家路由强制轮询（完美负载均衡）时每 GPU 416 TFLOP/s（16.7% MFU）；更长上下文和该规模的多模态模型尚未验证。

## PLE：词汇特征与分片

PLE 用原始 token ID 寻址其表，独立于隐藏激活和多模态旋转坐标。恰好一个解码器层拥有 PLE。八个 bigram 和八个 trigram 哈希头各检索一个 160 宽向量：

```text
token IDs [T]
  -> document-local n-grams
  -> hash IDs [T,16]
  -> table lookup [T,16,160]
  -> lexical features [T,2560]
```

表为 `[320001536,160]`：51.2B 参数。每个哈希头使用自己的行范围。128 个编号 HF 分片是检查点分区，不决定训练 owner 数量。

PLE 将词汇特征投影为键 `[T,4,2560]` 和值 `[T,2560]`。每条隐藏流提供一个查询；归一化的查询/键分数产生门 `[T,4]`。门控值形成 `[T,4,2560]`，经过膨胀深度卷积，然后加到残差流。

权重是稠密 SPMD 网格（`dp`、`cp`、`tp`）上的 DTensor，排除在 FSDP all-gather 之外。每个 DP owner 保留连续行（`Shard(0)`）；TP 下，其 TP rank 拆分这些行的特征（`Shard(1)`）。TP rank 持有相同 token ID，因此只有 DP rank 交换 ID 和向量（本地特征宽度），然后 TP 收集查找到的特征。
预处理对哈希 ID 去重，并打包进每个 owner 一个固定大小的 `C` ID 槽；前向交换请求的向量，反向将梯度返回给 owner。固定槽保持 all-to-all 形状静态，因此 `C` 必须覆盖任何数据的最坏情况。

每个 token 从 16 个哈希头各读一行，因此一个 rank 最多请求 `16*T` 个 ID。该总量是理想值：一个 rank 可能需要全部 `16*T`，因此任何固定大小方案都无法发送更少。

简单选择是每个 owner 一个 `16*T` 槽，总是安全但发送 `P*16*T`，即理想的 `P` 倍（`P` = DP 度）。我们改用更紧的槽大小 `C`。每个头拥有连续行范围，因此一个 owner 从每个头最多收到 `min(T, 其拥有的头的行数)` 个 ID。`C` 是各头之和，在所有 owner 上取最大。它只依赖 `T` 和表布局，因此每个 rank 无需通信即可算出相同 `C`。

开销为 `P*C / (16*T)`，即相对理想的流量：1.0 为理想，简单 `16*T` 槽会给出 `P`。

| DP | `T=4096` 时 `C` | 开销 | `T=16384` 时 `C` | 开销 |
| --- | ---: | ---: | ---: | ---: |
| 8 | 8,604 | 1.05 | 33,180 | 1.01 |
| 16 | 4,508 | 1.10 | 16,796 | 1.03 |
| 64 | 4,508 | 4.40 | 16,796 | 4.10 |

每个头覆盖约 1/16 的表。DP16 以内一个 owner 持有一或两个头，因此槽保持在理想值的 10% 以内。超过 DP16 后多个 owner 拆分一个头，每个仍为其保留 `T` 个槽，因此开销随 `P` 增长。

PLE 需要固定行所有权和单向量查找。All-to-all 路由和本地 `F.embedding` 满足这些需求，同时复用 TorchTitan 的 DTensor、优化器、裁剪和检查点，集成工作量少于 TorchRec 更广泛的嵌入栈。

```text
8-row table, 2 DP owners:
  rank 0 owns rows 0..3; rank 1 owns rows 4..7
  requests [6,1,6] -> unique requests [1,6]
  row 1 goes to rank 0; row 6 goes to rank 1
  returned vectors are restored as [E6,E1,E6]
```

Adam、裁剪和 DCP 使用原生 DTensor 支持；HF 转换避免收集全表。FP32 参数、梯度和 Adam 状态在 DP=8 时每个 owner 花费 95.37 GiB，不含主干和激活。
详见 [ROW_SHARDED_EMBEDDING.md](ROW_SHARDED_EMBEDDING.md)。

哈希历史在 EOS 后重置。位置零的文档边界重置哈希、PLE 卷积和 QSA 可见性。

## 优化器

[优化器配方](../../../torchtitan_recipes/models/qwen3_8_flash_next.py) 实现
[技术报告第 3.1 节](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/4f58f4ddd855bedaf7eadcd53bbbb1b3362cecff/tech_report.pdf#page=16) 的参数分配：

| 优化器 | 参数 |
| --- | --- |
| DistMuon | 注意力 Q/K/V 和输出投影、GDN Q/K/V 和输出投影、路由/共享专家 gate/up/down 矩阵、PLE 键/值投影 |
| AdamW | Token 嵌入、输出头、MoE 路由器、GR 低秩投影、注意力/GDN 输出门、GDN 衰减/beta、归一化和卷积 |
| Adam，无权重衰减 | PLE 查找表 |

DistMuon 使用 Nesterov 动量 0.95、八次
[Polar Express 迭代](https://github.com/liangyuwang/FSDP-Canzona/blob/426e8c50363cc3401e64eeb80c9e6e2451ea2ed4/matrix_based_optimizer/optimizers/muon.py)、
epsilon `1e-14` 和 `match_rms_adamw` 缩放。QKV 和门使用单独的 `Linear` 参数；`BlockShard` 按头正交化 Q/K/V。
多模态配方将视觉保留在 AdamW 上。

## 支持的并行

| 特性 | 说明 |
| --- | --- |
| FSDP | 逐层主干分片；PLE 持久行所有权 |
| TP / SP | QSA 头和 PLE 特征列；度必须整除 2 个 QSA KV 头 |
| EP | 现有路由专家放置；默认 all-to-all 调度器，125B 配方和 B200 图测试中使用带加速器图的 Dist-MoE |

EP 独立于 PLE 分片专家：FSDP8/EP4 时每个 rank 拥有八分之一 PLE 行和四分之一专家，每个专家分片内使用 FSDP2。PLE 的 CPU 卸载和延迟 FSDP 归约不受支持。PP 和 CP 在配置时被拒绝。

## 数值一致性与测试

QSA 注意力在 CUDA 上使用 Attention Gym 的融合 `gather_attn` 内核。分离值和分组 KV 头时，它仅支持 BF16 下的 SM100/SM103（Blackwell）；其他 GPU 在 Attention Gym 中报错。因此 GPU 集成测试在 B200 套件中运行（包括一个 FSDP2 + EP2 Dist-MoE 测试，将前向/反向捕获并重放为一个加速器图），CPU 测试使用 Attention Gym 的参考内核。

缩减模型使用相同权重的结果，在 GB300 上用
[HF 一致性脚本](../../../scripts/checkpoint_conversion/numerical_tests_qwen3_8_flash_next.py) 测量：

| 对比 | 最大输出误差 | 参数梯度余弦 |
| --- | ---: | ---: |
| 文本，BF16 | `4.64e-3` | `0.999926` |
| 视觉，FP32 | `3.35e-7` | `>0.999999999999` |
| 图像/视频，BF16 解码器 + FP32 视觉 | `3.17e-3` | `0.999953` |

两个 BF16 模型的舍入与精确结果不同，因此脚本以 HF 模型的 FP32 副本为基准门控 logits：torchtitan 的最大误差必须保持在 BF16 HF 参考的两倍以内。实测：文本 `4.03e-3` 对 `4.28e-3`，图像/视频 `4.34e-3` 对 `4.59e-3`。表中的输出误差是相对 BF16 HF 参考，仅供参考。

PLE 与 GDN 层一样，不跳过文档内的填充；数据流水线从文档位置 0 开始尾部填充。QSA 从其块中排除填充。

不间断训练按位可复现。恢复训练不可复现，因为视觉编码器中共享的编译 `flex_attention` 根据进程历史选择静态或动态形状
（[#6093](https://github.com/pytorch/torchtitan/issues/6093)）。

## TODO

- 共享教师落地后，用
  [共享教师](https://github.com/pytorch/torchtitan/pull/5863) 替换模型本地索引器教师。
- 在 16K token、16-GB300 微调配方之外验证发布模型：表卸载、更长上下文注意力和多模态模型。