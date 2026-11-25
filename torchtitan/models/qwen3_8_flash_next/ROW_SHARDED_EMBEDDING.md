# 行分片嵌入查找

`row_sharded_embedding.py` 使用 PyTorch 函数式集合通信和 `F.embedding`，将单个嵌入请求路由到固定的行所有者。

## 路由与形状

调用方提供一个命名的一维 `DeviceMesh`；每个 rank 拥有相等的连续行块，因此全局行数必须能被 mesh 大小整除。每个 rank 使用相同的权重 dtype 和特征宽度。输入 ID 可以是 int32 或 int64，形状任意。

```python
from torchtitan.models.qwen3_8_flash_next.row_sharded_embedding import (
    prepare_row_sharded_embedding,
    row_sharded_embedding,
)

# Once per microbatch, before model forward. max_ids_per_owner bounds the
# distinct IDs any requester sends to any one owner, and must be equal on
# every rank of row_mesh.
routing = prepare_row_sharded_embedding(
    input_ids,
    num_embeddings=num_global_rows,
    row_mesh=row_mesh,
    max_ids_per_owner=max_ids_per_owner,
)

# Forward: local master rows, selected vectors returned in the compute dtype.
embeddings = row_sharded_embedding(weight_RD, routing, dtype=torch.bfloat16)
```

每个请求方向每个所有者发送固定 `C = max_ids_per_owner` 个槽位的不同 ID，因此形状和 all-to-all 切分不依赖具体 ID。这样查找可被 CUDA graph 捕获，且计数无需回到主机。

以两所有者上的八行表、`C=2` 为例：所有者 0 存储行 0-3，所有者 1 存储行 4-7。请求 `[6,1,6]` 变为排序去重后的 ID `[1,6]`，已按所有者分组，填充槽位 `[[1,-],[6,-]]`。所有者 0 读取本地行 1；所有者 1 读取本地行 2；未使用的槽位读取本地行 0，但请求方不会选中它们。返回向量按原始请求顺序展开为 `[E6,E1,E6]`。

| 阶段 | 张量与通信 |
| --- | --- |
| 准备 | 排序去重 ID，交换 `[P,S]` 请求槽位（`S = C + 1`：`C` 个 ID 加请求方的有效标志）。所有者收到本地行索引 `[P*C]`。 |
| 前向 | 从本地权重读取 `[P*C,D]` 向量，转换类型，以等分 all-to-all 返回，展开重复项。 |
| 反向 | 在请求方汇总重复梯度，反向交换向量，累加到所有者本地行梯度。未使用槽位梯度为零。 |

结果形状为 `[*input_ids.shape, D]`，其中 `D` 为本地特征宽度。每次交换每 rank 移动 `P*C` 个 ID 或向量，因此紧凑的 `C` 使流量接近实际请求的向量数。等分大小由各 rank 的本地 `C` 决定，因此 `row_mesh` 的每个 rank 必须传入相同值；不一致会导致交换大小错误而非报错。`C` 应来自各 rank 一致的量（表布局和配置的每 microbatch token 数），绝不能来自本地 ID。例如，Flash-Next 的哈希头各自拥有连续行范围，因此一个所有者从每个命中其行的头最多收到每 token 一个 ID。

## 精度与优化器语义

只有被选中的向量会转换为请求的计算 dtype。使用 FP32 主权重和 BF16 计算时，重复梯度在每个请求方以 BF16 累加，然后在所有者处以 FP32 求和。这与混合精度嵌入的本地累加一致：`[256,1]` 先舍入为 256，再加另一请求方的 -256，结果为零。

集合通信求和梯度时不做 mesh 大小除法。调用方归一化损失，并在累积的 microbatch 间保持权重不变。查找是一个 `autograd.Function`，其反向仅执行反向 all-to-all 并在本地构建 DTensor 梯度；优化器更新在反向之后进行。梯度和优化器动量保持为稠密本地分片。

## 参数与重计算契约

调用方分配本地行、构建权重的 DTensor，并将该参数排除在 FSDP 聚合之外。权重是在行轴上 `Shard(0)` 分片的 DTensor；其 mesh 可携带更多轴（Flash-Next 使用稠密 SPMD mesh）。其梯度保留权重的放置方式，因此现有裁剪、优化器和原生 DCP 可直接消费。

准备阶段不执行设备到主机的同步。准备好的元数据可在重计算期间复用，且是注册的 pytree，因此 CUDA graph 重放会刷新其张量。每个所有者都参与，包括没有输入 ID 的 rank。每个请求方的槽位携带有效标志；表外 ID 或某所有者超过 `C` 个不同 ID 会在请求交换后于每个所有者上触发异步断言失败。

所有所有者必须一起重放响应集合通信。非重入的 PyTorch checkpointing 必须禁用提前停止，以免一个 rank 跳过另一 rank 重放的集合通信。

## 验证

四 rank CPU/Gloo 测试将值、累积梯度和优化器更新与稠密参考对比。用例包括重复 ID、BF16 抵消、无输入 ID 的 rank、整数 dtype、重计算，以及满和未使用的请求槽位。无效 ID 和容量溢出会在每个所有者上失败。

```bash
pytest tests/unit_tests/cpu/models/qwen3_8_flash_next/test_row_sharded_embedding.py -q
```