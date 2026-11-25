# FlexShard

FlexShard 提供 PyTorch 原生的构建模块，用于以不同于持久化 DTensor 参数存储布局的布局运行优化器计算。它规划打包的存储到计算重分布，并将通信与优化器工作重叠。DistMuon 是其首个使用方。

## 公共 API

公共 API 从 `torchtitan.distributed.flex_shard` 导出：

- `ComputeLayout` 描述在命名 `DeviceMesh` 轴上的临时优化器计算分片，使用 PyTorch DTensor placements 加上 `BlockShard` 或 `Owned`，并可选地指定多个轴对一个张量维度分片的顺序。
- `ComputeLayout.compute_dtype` 是该参数优化器计算所用的 dtype；`None` 保持存储 dtype。DistMuon 以 BF16 运行 Newton-Schulz，且只接受 `torch.bfloat16`，因此每个 FP32 参数都会设置它。输入以该 dtype 发送到计算 rank，方向也以该 dtype 发回。
- `BlockShard` 沿一个张量维度对完整的连续块进行分片。它保留张量的 rank 和全局形状，且从不创建张量视图。
- `Owned` 将一个完整的子组本地逻辑张量分配给计算阶段动态选定的一个 rank。
- `BucketConfig` 按完全限定名对参数分组和排序，用于打包重分布和通信-计算重叠。
- `BlockShard.block_sizes` 是一个非空元组，描述可独立分片的块大小的重复序列。均匀块使用 `(R,)`。例如，`(128, 64)` 将 384 行划分为四个块：128、64、128 和 64 行。块按数量分配，因此各 rank 可以拥有不同的行数。DistMuon 对每个块独立运行 Newton-Schulz 和宽高比学习率调整。Kimi 的共享 `wkv_a` 投影使用 `(512, 64)` 分别分配其 KV 潜在矩阵和 RoPE 键矩阵。
- `DistMuon` 在 `compute_sharding_by_fqn` 中消费与优化器无关的每参数 `ComputeLayout` 值。DistMuon 的 `BlockShard` 路径只接受具有连续本地 DTensor 存储的 2D 参数。placement 必须针对张量维度 0；前导维度必须非零且可被 `block_sizes` 之和整除。
  使用 `block_sizes=(R,)` 时，每连续 `R` 行形成一个独立的 `[R, C]` 矩阵。多个块大小允许这些矩阵具有不同的行数。FlexShard 使用块边界路由扁平的 2D 计算张量。DistMuon 的规划代码直接根据每个 rank 的计算形状和逻辑起始行构造零拷贝的跨步 `[M, R, C]` 视图。对于 `(R,)`，该视图等价于展开行维度。
  原生矩阵批次 `[..., R, C]` 使用 `Shard(0)` 分配其最外层批次维度，或使用 `Owned` 将完整批次分配给一个 rank。不使用 `BlockShard` 的单个 2D 矩阵使用整矩阵计算，例如 `Owned`。构建器验证命名 DTensor 参数并规划其存储到计算的转换。

存储 placements 仅描述持久化所有权；它们不定义 Muon 矩阵边界。扁平矩阵批次计算在至多一个非单位 mesh 轴上支持 `BlockShard`。该轴上的存储可以使用精确的 `Shard(0)` 或 `Replicate`；其他每个非单位存储 mesh 轴都必须复制。

原生 `[..., R, C]` 参数可以将 `Replicate()`、矩阵行维度的分片或矩阵列维度的分片重分布到一个 mesh 轴上的 `Shard(0)` 计算，其他每个非单位存储 mesh 轴都复制。

多个 mesh 轴可以对同一张量维度分片。默认按存储 mesh 顺序应用；`shard_order_by_tensor_dim` 指定不同顺序，最外层轴在前。例如，保留 EP 轴的 `Shard(0)`，同时将其本地专家域在前置 `edp_shard` 轴上重新分区，使用两个轴上的 `Shard(0)` 并设置 `shard_order_by_tensor_dim={0: ("ep", "edp_shard")}`。FlexShard 从绑定的 mesh 推导每个轴的拆分因子，然后将 `edp_shard` placement 降级为子组本地 `Shard(0)` 以供优化器执行。

计算分片是构建时配置。它在优化器构建时被验证并冻结，但不存储在状态字典中；检查点恢复必须使用匹配的值重建优化器。

## TorchTitan Kimi 集成

[Kimi 测试配方](../../../torchtitan_recipes/tests/models/kimi_k2_7.py) 提供了首个 TorchTitan 集成。其共享优化器配置：

- 从注意力、稠密 MLP、路由和共享专家以及路由器中选择矩阵参数供 DistMuon 使用。其他参数继续使用 AdamW。
- 定义每个选中参数的计算布局，包括对兼容注意力投影的逐头 Muon。
- 将层分组为桶，使计算就绪的工作能与打包重分布重叠。

相同的构建方式可用于 Kimi 系列、Kimi-VL 和 Moonlight 配方。

## 包边界

FlexShard 目前位于 TorchTitan 中，其 API 尚在成熟中。我们打算将此目录并入一个独立的 Python 包和仓库。

保持此目录自包含且仅依赖 PyTorch。依赖应单向流动：

```text
TorchTitan 组件和模型 -> FlexShard -> PyTorch
```

FlexShard 不得依赖 TorchTitan 模型或训练基础设施。