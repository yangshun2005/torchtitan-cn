# DeepSeek-V3 MTP 设计说明

本文档描述 torchtitan 中当前的 DeepSeek-V3 多 Token 预测（MTP）实现，涵盖 MTP 模块的配置与集成方式、CP/TP/SP 下的输入处理、FSDP/CP/TP/EP 下的模块分片，以及 MTP 损失的计算方式。目标是先对齐当前实现边界，再明确列出剩余的 TODO 项。

## 1. MTP 配置与集成

### 1.1 配置入口

DeepSeek-V3 MTP 目前通过模型配置构建器启用：

- `build_model_config(..., num_mtp_layers=N)` 控制是否启用 MTP。
- `_build_mtp_layers(...)` 基于最后一个主解码器层配置构建 MTP 层配置。
- `deepseek_v3_debugmodel_mtp()` 是当前的调试配置示例。它设置 `num_mtp_layers=1`，并使用 `ChunkedLossWrapper` 搭配 `MTPCrossEntropyLoss.Config(mtp_scale=0.3)` 作为内层损失。

每个 MTP 层沿用普通解码器层结构，并包含以下 MTP 专用模块：

- `enorm`：对当前 MTP 深度移位的 token 嵌入做归一化。
- `hnorm`：对上一隐藏状态做归一化。
- `eh_proj`：将拼接后的嵌入分支和隐藏分支投影回隐藏维度。
- `mtp_norm`：在输出投影路径前对每个 MTP 深度的输出做归一化。

### 1.2 模型集成

`DeepSeekV3Model` 继承自 `MTPDecoder`（`torchtitan/models/common/mtp.py`），DeepSeek-V4 也共享该类。`MTPDecoder` 负责 MTP 输入预处理、FSDP 包装、remat 容器和 FLOPs；各模型自行实现 MTP 块和 MTP 前向。若 `mtp_layers` 为空，`MTPDecoder` 和 `DeepSeekV3Model.forward()` 回退到普通 `Decoder` 路径，保持非 MTP 行为不变。

当 `mtp_layers` 非空时，执行分为两个阶段：

1. `MTPDecoder.preprocess_inputs()` 在通用 CP/SPMD 输入处理之前，为每个 MTP 深度准备一对移位的 token/label。
2. 前向先在原始 token 上运行主解码器，然后每个 `DeepSeekV3MTPBlock` 消费其准备好的移位 token 和上一深度的隐藏状态。

数据加载器无需为 MTP 扩展输入序列。模型的输入预处理在前向之前从原始批次中推导出额外的 token/label 对。

### 1.3 当前实现细节

`MTPDecoder.Config` 存储：

- `mtp_layers`

`num_mtp_layers` 仍是 `build_model_config` 的参数，但不存储在 `MTPDecoder.Config` 上。实际 MTP 深度数由 `len(mtp_layers)` 推导。

模型构造函数在构建两组模块之前，根据当前 `ParallelismContext` 为普通解码器层和 `mtp_layers` 配置分片。

MTP 通过在 CP 分片之前构造移位输入来支持 CP。PP 仍不支持，因为其阶段归属和通信路径尚未集成。

## 2. MTP 输入处理

### 2.1 主模块输入

主模块使用普通训练器输入：

- `tokens`
- `positions`
- `attention_metadata`

沿用论文图中的记号，若主模型当前训练窗口为 `t1 ... t4`，预处理会构造对齐的 `t2 ... t5` 和 `t3 ... t6` token/label 对供各 MTP 深度使用，而不扩展数据加载器序列。

### 2.2 `roll_mtp_sequence`

MTP 移位 token 和移位 label 由 `roll_mtp_sequence()` 生成。该函数沿序列维度移位：

- `shift=1` 时，位置 `i` 读取原始序列中的 token `i+1`。
- `shift=k` 时，位置 `i` 读取原始序列中的 token `i+k`。
- 超出当前序列长度的位置用 `fill_value` 填充。
- 提供 `positions` 时，若移位后的位置跨越打包文档边界，则标记为无效。

该函数可选返回 `valid_mask`。掩码表示当前位置及其移位来源是否均为同一文档中的非填充位置。在无效位置，MTP 块会屏蔽上一隐藏状态分支，而 MTP 损失通过 `IGNORE_INDEX` 忽略对应 label。已设为 `IGNORE_INDEX` 的 label（如被掩码的 prompt 目标）只影响预测损失，不会使原本有效的 token 失去 MoE 路由资格。

例如，考虑一个包含两个文档的较长打包批次行：

- `sequence=[A0, A1, A2, B0, B1, B2, B3]`
- `positions=[0, 1, 2, 0, 1, 2, 3]`

`shift=1` 时，`roll_mtp_sequence()` 返回 `[A1, A2, fill, B1, B2, B3, fill]`。前两个位置留在文档 A 内，位置 `B0`、`B1`、`B2` 留在文档 B 内。`A2` 之后的槽位被填充，因为下一个物理 token 是 `B0`，其位置 id 从 `2` 重置为 `0`，使用它会跨越打包文档边界。最后一个槽位被填充，因为序列中没有下一个 token。对于 MTP 输入 token，`fill` 为 token id `0`；对于 MTP label，`fill` 为 `IGNORE_INDEX`。

### 2.3 各深度的输入与 Label 对齐

训练器仍提供普通的下一 token 预测批次。对于未打包的行：

| 路径 | 模型输入 token | 损失 label |
| --- | --- | --- |
| 主模型 | `[t0, t1, t2, t3, t4, t5]` | `[t1, t2, t3, t4, t5, t6]` |
| MTP 深度 1 | `[t1, t2, t3, t4, t5, FILL]` | `[t2, t3, t4, t5, t6, IGNORE]` |
| MTP 深度 2 | `[t2, t3, t4, t5, FILL, FILL]` | `[t3, t4, t5, t6, IGNORE, IGNORE]` |

> **注意：** 此对齐方式与 DeepSeek-V3 论文图 3 不同。
> 该图展示每个 MTP 深度具有等宽的连续窗口，随着输入窗口移位，有额外的未来 token 可用。TorchTitan
> 的数据加载器提供一对固定长度的输入/label，因此本实现在同一输入行内移位，并填充不可用的尾部。
> 因此每增加一个 MTP 深度，有效目标就比上一深度少一个。

对于打包行，每个文档独立移位。考虑：

- `tokens=[A0, A1, A2, B0, B1, B2, B3, C0, C1, C2]`
- `positions=[0, 1, 2, 0, 1, 2, 3, 0, 1, 2]`

| 移位 | 结果 |
| --- | --- |
| `shift=1` | `[A1, A2, FILL, B1, B2, B3, FILL, C1, C2, FILL]` |
| `shift=2` | `[A2, FILL, FILL, B2, B3, FILL, FILL, C2, FILL, FILL]` |

MTP 输入 token 和 MTP label 使用同一辅助函数，但填充值不同。输入 token 填充使用 token id `0`；label 填充使用 `IGNORE_INDEX`。

MTP 块接收原始 `positions`，而非移位后的 positions。这是有意为之：每个输出隐藏状态在注意力/RoPE 中仍锚定在其原始目标位置，而移位嵌入分支贡献当前 MTP 深度所选 token 的内容。移位 `positions` 会错误地将输出移动到移位 token 的位置。

### 2.4 与 DeepSeek-V3 论文的行为对比

本 PR 遵循论文风格的 MTP 架构：每个 MTP 深度将同一序列位置上上一深度的隐藏状态与该深度移位的 token 嵌入相结合。这与仅移位损失的设计不同——后者每个预测头消费相同的隐藏/输入状态，只移位其 label。

与简化论文图不同的输入窗口对齐和打包文档处理见第 2.2 和 2.3 节。Megatron-LM 使用类似的 `roll_tensor` 操作在每个 MTP 深度移位 token 输入。

参考：

- [multi_token_prediction.py](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/multi_token_prediction.py#L137)


### 2.5 TP

TP 下 token id 通常被复制。MTP 输入预处理在 CP 分片和 SPMD 标注之前移位 token id，然后解码器对每个准备好的输入调用共享嵌入模块：

```python
mtp_input_tokens, mtp_input_valid_mask = roll_mtp_sequence(
    tokens,
    shift=depth,
    positions=positions,
    fill_value=0,
    return_valid_mask=True,
)
mtp_input_embed = self.tok_embeddings(mtp_input_tokens)
```

这样嵌入输出遵循现有的嵌入分片配置，而无需在模型代码中手动移位隐藏状态。

### 2.6 SP

启用 SP 时，隐藏状态沿序列维度分片。此时复用主嵌入输出并应用局部移位是错误的，因为局部 shard 无法表示跨 shard 边界的全局序列移位。

因此当前实现：

1. 在 token-id 级别构造 `mtp_input_tokens`。
2. 复用 `tok_embeddings(mtp_input_tokens)`，使移位嵌入输出遵循配置的分片策略。
3. 将 `mtp_input_valid_mask` 传入 `DeepSeekV3MTPBlock`；SP 下块级 `ShardingConfig` 将该掩码对齐到激活布局。

换言之，SP 处理不在前向代码中显式调用 `redistribute`。掩码输入的放置转换由 MTP 层分片配置描述。

## 3. MTP 并行化与分片

### 3.1 FSDP

`MTPDecoder._apply_fsdp()` 临时将 `mtp_layers` 插入 `model.layers`，并调用 `Decoder._apply_fsdp()`。这使 MTP 层获得与普通解码器层相同的 FSDP 包装、专家包装和重分片策略。

好处是 MTP 复用现有 DeepSeek-V3 FSDP 策略，无需复制完整包装逻辑。缺点是实现较为间接。未来清理可将通用 FSDP 应用逻辑抽取为普通解码器层和 MTP 层共享的辅助函数。

### 3.2 TP/SP

MTP TP/SP 分片在 `flavors.py` 的 `_build_mtp_layers()` 构建每个 MTP 块时设置。主要规则：

- MTP 注意力/前馈/MoE 子模块复用普通 DeepSeek-V3 层分片策略。
- `enorm`、`hnorm` 和 `mtp_norm` 使用与激活布局匹配的 norm 分片。
- `eh_proj` 的输入和输出激活与稠密激活放置对齐。
- 启用 SP 时，`mtp_input_valid_mask` 通过块级 `ShardingConfig` 从解码器 token-ID 放置（DP/CP 分片且 TP 复制）重分布到 1D 序列并行 token-ID 放置。

这使 MTP 块前向代码独立于显式重分布细节。放置由分片配置描述。

### 3.3 EP

若 MTP 层包含 MoE，当前实现复用普通 DeepSeek-V3 MoE 分片配置：

- 专家参数使用 EP 相关放置。
- 分组专家参数名来自内层专家初始化配置。
- 路由器、共享专家和 token 分发/合并路径遵循普通层配置。

预期行为是 EP 下的 MTP 层与普通解码器层一致，而非为 MTP 引入单独的专家并行语义。

### 3.4 上下文并行

MTP 预处理在通用 CP 分片步骤之前构造移位 token、label 和有效性掩码。同一负载均衡器随后将主模型和 MTP 张量一起分片，因此对应局部位置保持对齐，无需在 CP rank 之间交换边界。

## 4. MTP 损失处理

### 4.1 输入形式

`MTPDecoder.preprocess_inputs()` 返回对齐的元组，包含主模型和各深度的 token 输入与 label。`DeepSeekV3Model.forward()` 返回相同顺序的预测元组。普通路径上预测为 logits；`ChunkedLossWrapper` 的 `_skip_lm_head=True` 时为隐藏状态。

### 4.2 主损失

主模型损失使用元组索引零处的原始 label：

```python
main_loss = cross_entropy(pred[0], labels[0])
```

这使主训练目标与普通解码器路径保持一致。

### 4.3 MTP 辅助损失

输入预处理将 MTP 深度 `depth` 的移位 label 存储在匹配的元组索引处。损失消费已对齐的对：

```python
mtp_loss = cross_entropy(pred[depth], labels[depth])
```

`IGNORE_INDEX` 屏蔽超出当前序列范围、跨越打包文档边界、落在填充上或被数据集损失掩码排除的 label。每个 MTP 损失项权重为 `mtp_scale / num_mtp_layers`；加权项求和等价于对 MTP 损失取平均后乘以 `mtp_scale`：

```python
total_loss = main_loss + mtp_scale * mean(mtp_losses)
```

预处理还产生单独的目标 token 计数和路由 token 计数向量，顺序为 `(main, depth 1, ..., depth N)`。训练器在优化器步内对两个向量求和并一起归约。每个预测损失使用其目标计数，每个 MoE 辅助损失使用其所属块的路由 token 计数。这使指令式 label 掩码与 MoE 路由相互独立。

### 4.4 损失并行

配置 `global_vocab_size` 时，`MTPCrossEntropyLoss` 使用损失并行交叉熵路径所需的词表大小。主预测和辅助预测共享同一交叉熵辅助函数，因此各深度的损失语义不会分歧。

### 4.5 分块损失

`ChunkedLossWrapper` 接受一对预测/label 张量，或对齐的预测和 label 元组。对每个分块，它对每个预测应用共享 LM head，调用内层损失时保持张量或元组结构，并将一个累积的隐藏状态梯度路由回每个模型输出。`MTPCrossEntropyLoss` 接收对齐元组，并组合主交叉熵目标和加权辅助交叉熵目标。

## TODO

### 1. PP 集成

PP 下必须定义 MTP 层在各流水线阶段间的归属，并将辅助 logits/损失路由到损失阶段。仍需完成：

- 哪个 PP 阶段拥有 MTP 层。
- 主隐藏状态和 MTP 隐藏状态如何跨阶段传输。
- 多深度 MTP 输出如何传递到损失阶段。
- PP 与 FSDP、TP、EP 组合时的包装和通信顺序。

在 PP 集成完成之前，应继续不支持 PP 下启用 MTP。