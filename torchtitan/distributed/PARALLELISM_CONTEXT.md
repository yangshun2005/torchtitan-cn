# ParallelismContext 网格轴

[`ParallelismContext`](parallelism_context.py) 构建训练所用的 `DeviceMesh` 视图。轴名称定义在 `MeshAxisName` 中，度数来自 `ParallelismConfig`。

## 轴

| 轴 | 配置字段 | 作用 |
|------|--------------|------|
| `pp` | `pipeline_parallel_degree` | 流水线阶段 |
| `dp_replicate` | `data_parallel_replicate_degree` | DDP / HSDP 复制 |
| `dp_shard` | `data_parallel_shard_degree` | 稠密网格上的 FSDP 分片 |
| `dp` | 派生：`dp_replicate * dp_shard` | 稠密前向/反向网格上折叠的 DP 轴 |
| `cp` | `context_parallel_degree` | 上下文（序列）并行 |
| `tp` | `tensor_parallel_degree` | 张量并行 |
| `ep` | `expert_parallel_degree` | 稀疏网格上的专家并行 |
| `edp_shard` | 派生：`dp_shard * cp * tp // ep` | EP 区域内的 FSDP |

`data_parallel_shard_degree=-1` 会在其他稠密度数之后填充剩余 rank：`world_size // (dp_replicate * cp * tp * pp)`。

## 为什么 EP 不在 world_size 乘积中

`ParallelismContext` 检查

```text
dp_replicate * dp_shard * cp * tp * pp == world_size
```

EP 不是因子，因为它不增加 GPU。稠密乘积已覆盖每个 rank。EP 是对同一个 1D world 网格的第二次 unflatten，让路由专家可以重新分组这些 rank。

## EP 如何划分

`build_mesh` 从 1D world 网格出发，unflatten 出三个视图（最后一轴最快）：

| 视图 | 轴 | 度数 |
|------|------|---------|
| 稠密存储 | `pp, dp_replicate, dp_shard, cp, tp` | 配置度数 |
| 稠密前向/反向 | `pp, dp, cp, tp` | `dp = dp_replicate * dp_shard` |
| 稀疏 | `pp, dp_replicate, edp_shard, ep` | `edp_shard = dp_shard * cp * tp // ep` |

`pp` 和 `dp_replicate` 在每个视图中都保持在外层。内部稠密区域 `dp_shard * cp * tp` 是 EP 借用的池：

```text
ep 必须整除 (dp_shard * cp * tp)
```

等价于 `dp_shard * cp * tp == edp_shard * ep`。稠密模块（注意力、共享专家、嵌入）留在稠密网格上。路由专家使用稀疏网格：专家权重在 `ep` 上 `Shard(0)`，剩余 FSDP 在 `edp_shard` 上。放置方式见 [MoE 分片](../models/common/moe/SHARDING.md)。

## 示例

取 `tp=2`、`cp=2`、`dp_shard=2`、`pp=1`、`dp_replicate=1`。则 `world_size = 8`。内部稠密区域为 `2 * 2 * 2 = 8`，因此 `ep=4` 和 `ep=8` 都有效。省略大小为 1 的外层轴，rank `0..7` unflatten 为：

稠密 `(dp_shard, cp, tp)`：

| Rank | `dp_shard` | `cp` | `tp` |
|------|------------|------|------|
| 0 | 0 | 0 | 0 |
| 1 | 0 | 0 | 1 |
| 2 | 0 | 1 | 0 |
| 3 | 0 | 1 | 1 |
| 4 | 1 | 0 | 0 |
| 5 | 1 | 0 | 1 |
| 6 | 1 | 1 | 0 |
| 7 | 1 | 1 | 1 |

`ep=4` 的稀疏视图（`edp_shard = 8 // 4 = 2`）：

| Rank | `edp_shard` | `ep` |
|------|---------|------|
| 0 | 0 | 0 |
| 1 | 0 | 1 |
| 2 | 0 | 2 |
| 3 | 0 | 3 |
| 4 | 1 | 0 |
| 5 | 1 | 1 |
| 6 | 1 | 2 |
| 7 | 1 | 3 |

在 Transformer 块上，这些视图意味着：

- 注意力（稠密网格）：FSDP 在 `dp_shard=2` 上（配对 `(0,4)`、`(1,5)`、`(2,6)`、`(3,7)`），CP 在 `cp=2` 上切分序列（配对 `(0,2)`、`(1,3)`、`(4,6)`、`(5,7)`），TP 在 `tp=2` 上分片投影（配对 `(0,1)`、`(2,3)`、`(4,5)`、`(6,7)`）。
- `ep=4` 的路由专家：每个 `edp_shard` 切片上 4 路专家并行（`(0,1,2,3)` 和 `(4,5,6,7)`）；剩余 FSDP 为 `edp_shard=2`（本例中与稠密 `dp_shard` 配对相同）。
- `ep=8` 的路由专家：`edp_shard=1`，因此每个 rank 持有不同的专家分片，EP 区域内没有剩余 FSDP。注意力不变；只有稀疏视图重新分组。

每种情况下都是同样的八块 GPU。提高 EP 不会增大 `world_size`。

## 保存布局

在 recipe 中设置 `config.parallelism.save_parallelism_folder = "parallelism"`，`ParallelismContext.from_config` 会构建网格，并让每个 rank `r` 写入 `<dump_folder>/parallelism/rank_<r>.json`。所有 rank 写入相同字段，不涉及集合通信。例如，rank 5 在 `pp=2`、`dp_shard=2`、`cp=2`、`ep=2` 下（此处为便于阅读做了缩进）：

```json
{
  "host": "node-b",
  "local_rank": 1,
  "global_rank": 5,
  "world_size": 8,
  "degrees": {"pp": 2, "dp_replicate": 1, "dp_shard": 2, "cp": 2, "tp": 1, "ep": 2},
  "meshes": {
    "loss": {"axis_names": ["loss_mesh"], "mesh": [4, 5, 6, 7]},
    "dense": {
      "axis_names": ["pp", "dp_replicate", "dp_shard", "cp", "tp"],
      "mesh": [[[[[0], [1]], [[2], [3]]]], [[[[4], [5]], [[6], [7]]]]]
    },
    ...
  }
}
```

`meshes` 保存该 rank 对 `_global_meshes` 中每个网格的视图：上述稠密存储和稀疏视图，以及从中切片的 `loss`、`spmd_dense_for_fwdbwd` 和 `spmd_sparse_for_fwdbwd` 网格。像 `dense` 这样的完整网格在每个文件中都相同。切片网格只包含该 rank 所在的子网格，因此上面的 `loss` 在 rank 4-7 上为 `[4, 5, 6, 7]`，在 rank 0-3 上为 `[0, 1, 2, 3]`；读取所有文件即可得到每个子网格。网格是全局 rank 的嵌套列表；沿某轴共享同一组的 rank，就是仅在该轴坐标上不同的那些 rank。