# 类型化决策（Jev 风格）微调

微调 Qwen3-1.7B-Base，使其在不生成文本的情况下，针对某个状态回答类型化问题，遵循 TypeSafe 的 Jev 方案：`choice`（分类）、`noul`（是/否）和 `score`（有序量表）。每个问题在一次前向传播中为每个选项给出一个概率，在答案槽位读取：

```text
Context:
<state>

Question (choice): Which team should handle this ticket?
Options:
A) Payments and refunds
B) Login and account access
C) Something else
Answer:  --one forward pass--> next-token logits
                               keep only " A", " B", " C"
                               softmax(logits / temperature)
                               -> {payments: 0.91, account: 0.06, other: 0.03}
```

训练只监督答案字母对应的 token。每个问题打包为独立文档，因此问题之间无法相互注意。

## 运行

```bash
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3-1.7B-Base --assets tokenizer safetensors config
python -m torchtitan_recipes.decision.prepare_data
python -m torchtitan_recipes.decision.evaluate --hf-assets ./assets/hf/Qwen3-1.7B-Base
NGPU=2 MODULE=torchtitan_recipes.decision.qwen3 CONFIG=qwen3_1_7b_decision ./run_train.sh --output-dir outputs/decision
python -m torchtitan_recipes.decision.evaluate --hf-assets ./assets/hf/Qwen3-1.7B-Base --checkpoint outputs/decision/checkpoint/exports/step-600
```

`prepare_data` 下载 [`avbiswas/bev-decision-150K`](https://huggingface.co/datasets/avbiswas/bev-decision-150K)，并将每个问题写为一个提示到 `./assets/hf/bev-decision-150K/`（约 247K 个训练问题）。带种子的打乱会在每个划分中随机化 `choice` 选项和 `noul` 的是/否顺序；`score` 等级保持有序。训练为 600 步全量微调。

支持的并行方式：默认 FSDP；在你自己的配置函数中向 `qwen3_1_7b_decision` 传入 `ParallelismConfig`，可启用张量并行（含损失并行）或上下文并行（Ulysses varlen attention）。不支持流水线并行，因为 Qwen3-1.7B 的嵌入层和输出层权重是共享的。

每行平均约 180 个 token，其中 1 个为监督 token。`AnswerTokenLoss` 仅在监督位置应用 `lm_head`；该配方还使用单一数据读取器、预取 16 个微批次、禁用激活重计算，并将每个微批次的文档数上限设为 256。在单台 GB300 上，与使用 `ChunkedLossWrapper`、默认重计算和数据加载、每微批次 512 个文档的同一配方相比，这些优化将吞吐量从 68.7K 提升到 105.5K tokens/s（80 步运行中第 21-80 步的均值）。

## 结果

1,500 个测试问题；温度在来自另外 250 个测试状态的问题上拟合。训练、校准和评估不共享任何状态。数值对为 基座 -> 微调后。

| 类型 | n | 准确率（越高越好） | Brier（越低越好） | ECE（越低越好） | 5% 误差下已回答比例（越高越好） |
|---|---|---|---|---|---|
| 全部 | 1500 | 0.611 -> 0.873 | 0.491 -> 0.177 | 0.062 -> 0.016 | 0.10 -> 0.79 |
| choice | 576 | 0.552 -> 0.889 | 0.528 -> 0.158 | 0.053 -> 0.019 | 0.25 -> 0.86 |
| noul | 609 | 0.783 -> 0.933 | 0.352 -> 0.097 | 0.103 -> 0.015 | 0.00 -> 0.96 |
| score | 315 | 0.384 -> 0.727 | 0.690 -> 0.369 | 0.086 -> 0.065 | 0.00 -> 0.34 |

"已回答"指按置信度从高到低取最大的问题比例，其错误率保持在 5% 或以下。这些是分布内结果：测试问题与训练来自相同源数据集。