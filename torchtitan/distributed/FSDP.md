# FSDP1 -> FSDP2

## 为什么需要 FSDP2？
PyTorch 的全分片数据并行（FSDP）API [`FullyShardedDataParallel`](https://pytorch.org/docs/stable/fsdp.html) 旨在提供高性能的 eager 模式实现，包括通信分桶和通信/计算重叠。它通过展平并拼接一组参数来定义 `FlatParameter`，以此表示一个通信桶。然而，这个 `FlatParameter` 使得对其中单个参数施加不同行为（如参数冻结、参数类型转换等）变得复杂，损害了可组合性，同时也使内部实现复杂化，例如状态字典逻辑多达数千行，且需要额外的通信。

鉴于这些限制，我们设计并实现了一个移除 `FlatParameter` 的 FSDP 重写版本。我们将此重写版本称为 FSDP2，原版本称为 FSDP1。FSDP2 面向与 FSDP1 相同的用例并有所扩展，同时仍致力于在 eager 模式下保持良好性能，采用了若干相同的技术。

与 FSDP1 相比：
- FSDP2 将分片参数表示为在 dim-0 上分片的 `DTensor`，便于对单个参数进行操作，实现无通信的分片状态字典，以及更简单的 meta 设备初始化流程。
- FSDP2 实现了改进的内存管理系统，通过避免 `recordStream` 实现更低且确定性的 GPU 内存占用，且无需任何 CPU 同步。

未来，FSDP2 将提供自定义 all-gather 的扩展点（例如用于 fp8 线性层的 fp8 all-gather），并改进 `torch.compile` 支持。

我们已使用 torchtitan 验证了 FSDP2 的数值正确性和性能（例如参见此 [PR](https://github.com/pytorch/torchtitan/pull/165)）。例如，在 8x H100 上运行某些 Llama-7B 任务时，FSDP2 实现了更高的 MFU，峰值内存比 FSDP1 低 7%，且损失曲线一致。

关于动机、API 和系统设计的更多细节，请参阅[此处](https://github.com/pytorch/pytorch/issues/114299)。在本 README 中，我们尽量提供更多面向用户的信息，减少系统设计细节。

## FSDP1 <> FSDP2 API 差异
我们介绍 FSDP1 和 FSDP2 之间的一些 API 差异。总体而言，我们希望最小化 API 表面（包括参数数量），以避免形成单体式 API。
```python
@contract(state_cls=FSDPState)
def fully_shard(
  module: nn.Module,
  *,
  mesh: Optional[DeviceMesh] = None,
  reshard_after_forward: Union[bool, int] = True,
  mp_policy: MixedPrecisionPolicy = MixedPrecisionPolicy(),
  offload_policy: OffloadPolicy = OffloadPolicy(),
) -> nn.Module:  # returns `module` for `contract` checks
```

| FSDP1 | FSDP2 |
| ----- | ----- |
| `module` | `module` |
| `process_group`/`device_mesh` | `mesh` |
| `sharding_strategy` | `reshard_after_forward` |
| `cpu_offload` | `offload_policy` |
| `auto_wrap_policy` | 已移除 |
| `backward_prefetch` | 已移除 |
| `mixed_precision` | `mp_policy` |
| `param_init_fn` | 已移除 |
| `device_id` | 已移除 |
| `sync_module_states` | 已移除 |
| `forward_prefetch` | 尚未实现 |
| `limit_all_gathers` | 已移除 |
| `use_orig_params` | 已移除 |
| `no_sync` | `set_requires_gradient_sync` |
| `ignored_modules`、`ignored_states` | `ignored_params` |

HSDP 梯度累积使用 `set_requires_all_reduce`（而非 `set_requires_gradient_sync`），因此复制组的 all-reduce 仅在最后一个优化器累积的微批次上运行。

- `fully_shard(module)` 类似于 `FullyShardedDataParallel(module)`，从 `module.parameters()` 中构建一个通信桶，但排除已分配给嵌套 `fully_shard`/`FullyShardedDataParallel` 调用的参数。
    - `fully_shard(module)` 在 `module` 上添加一个 `FSDPState` 对象，可通过 `fully_shard.state(module)` 访问，而不是作为 `nn.Module` 包装器。这通过 `@contract` 装饰器实现。
    - 对应用了 FSDP2 的 `model` 调用 `model.named_parameters()` 返回未改变的参数名和 `DTensor` 分片参数。这意味着优化器和梯度范数裁剪看到的是 `DTensor`。
    - `fully_shard(module)` 对 `module` 执行动态类替换。例如，如果 `type(module) is Transformer`，则 FSDP2 构造一个新类 `FSDPTransformer`，继承自 `FSDPModule` 和 `Transformer`，并将 `module.__class__` 设为 `FSDPTransformer`。这允许我们通过 `FSDPModule` 添加新方法和覆盖方法，而无需构造 `nn.Module` 包装器。
- FSDP1 的 `sharding_strategy` 和 `process_group`/`device_mesh` 映射到 FSDP2 的 `mesh` 和 `reshard_after_forward`。
  - `mesh` 对于 FSDP 应为 1D，对于 HSDP 应为 2D。对于 HSDP，我们假设第 0 个 mesh 维度为复制，第 1 个 mesh 维度为分片。如果 `mesh is None`，则 FSDP2 在默认进程组上初始化一个 1D 全局 mesh。
  - `reshard_after_forward=True` 或 `False` 决定参数在前向之后是否重新分片（释放）。如果为 `True`，则在反向时重新 all-gather。这在节省内存与额外通信之间进行权衡。
  - （实验性）`reshard_after_forward: int` 表示参数在前向之后重新分片到更小的 world size（例如 `reshard_after_forward=8` 可表示节点内），从而使反向 all-gather 在更小的 world size 上进行。
  - | FSDP1 | FSDP2 | DeepSpeed |
    | --- | --- | --- |
    | 1 个 `process_group` + `FULL_SHARD` | 1D `mesh` + `reshard_after_forward=True` | ZeRO-3 |
    | 1 个 `process_group` + `SHARD_GRAD_OP` | 1D `mesh` + `reshard_after_forward=False` | ZeRO-2 |
    | 2 个 `process_group`/2D `device_mesh` + `HYBRID_SHARD` | 2D `mesh` + `reshard_after_forward=True` | MiCS |
    | 2 个 `process_group`/2D `device_mesh` + `_HYBRID_SHARD_ZERO2` | 2D `mesh` + `reshard_after_forward=False` | - |
    | - | 1D/2D `mesh` + `reshard_after_forward=8`（`int`） | ZeRO++ hpZ |
- FSDP2 将 `mixed_precision` 映射到 `mp_policy`，将 `cpu_offload` 映射到 `offload_policy`。
  - 对于 `mp_policy`，我们移除了 `buffer_dtype`，将 `cast_forward_inputs` 和 `cast_root_forward_inputs` 简化为仅 `cast_forward_inputs`，并添加了 `output_dtype`。
  - 对于 `offload_policy`，我们添加了 `pin_memory` 选项以避免固定 CPU 内存。（此功能可能尚未合入。）
- FSDP2 移除了 `auto_wrap_policy`、`backward_prefetch`、`param_init_fn`、`device_id`、`sync_module_states`、`limit_all_gathers` 和 `use_orig_params`。
  - `auto_wrap_policy` 提供了一种语法糖，用于根据策略给定的谓词对模块调用 `FullyShardedDataParallel`，并将包装后的模块赋值给其父模块。FSDP2 不再是 `nn.Module` 包装器，因此无需将模块赋值回父模块。我们倾向于将此功能置于 `fully_shard` 之上，未来可能会提供类似 `auto_wrap_policy` 的工具。
  - FSDP2 始终遵循 `backward_prefetch=BACKWARD_PRE` 且不可配置，因为这是反向时正确重叠集合通信的唯一方式。`BACKWARD_POST` 在嵌套模块场景下可能[错误地](https://github.com/pytorch/pytorch/issues/108190)预取。
  - FSDP2 支持新的 meta 设备初始化流程，无需在分片*之前*在 GPU 上实例化模块，从而无需 `param_init_fn`。详见 [Meta 设备初始化](#meta-设备初始化)。
  - FSDP2 始终将托管的参数/缓冲区移动到 `mesh` 对应的设备，无需 `device_id`。例如，如果 `mesh.device_type` 为 `"cuda"`，则 FSDP2 使用当前 CUDA 设备。
  - FSDP2 使用新的内存管理系统，在保持通信/计算重叠的同时实现确定且更低的内存占用。该系统不需要任何 CPU 同步，因此无需 `limit_all_gathers`。
  - FSDP2 始终"使用原始参数"，因为不再有 `FlatParameter`，无需 `use_orig_params`。
- 如何在 FSDP2 中实现 `forward_prefetch` 仍在讨论中。

| FSDP1 | FSDP2 |
| ----- | ----- |
| `model.state_dict()`：完整状态字典 | `model.state_dict()`：分片状态字典（无通信） |
| `optim.state_dict()`：本地状态字典 | `optim.state_dict()`：分片状态字典（无通信） |
| `summon_full_params()` | 使用 `DTensor` API 如 `full_tensor()` |
| `FSDP.clip_grad_norm_()` | `nn.utils.clip_grad_norm_()` |
| `ShardedGradScaler` | `amp.grad_scaler.GradScaler` |


## Meta 设备初始化
之前使用 FSDP1：
```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
with torch.device("meta"):
    model = Transformer()
policy = ModuleWrapPolicy({TransformerBlock})
# Call `reset_parameters()` on every module
model = FSDP(model, auto_wrap_policy=policy)
# Call `param_init_fn` on every module
def param_init_fn(module: nn.Module) -> None: ...
model = FSDP(model, auto_wrap_policy=policy, param_init_fn=param_init_fn)
```
之后使用 FSDP2：
```python
with torch.device("meta"):
    model = Transformer()
for module in model.modules():
    if isinstance(module, TransformerBlock):
        fully_shard(module)
fully_shard(model)
for tensor in itertools.chain(model.parameters(), model.buffers()):
    assert tensor.device == torch.device("meta")
# Allocate buffers and sharded parameters on GPU
model.to_empty(device="cuda")
# Run user-defined initializers
model.init_weights() # or `model.apply(init_weights)`
```
FSDP1 要求在分片之前立即使用 `reset_parameters` 或 `param_init_fn` 将模块实例化到 GPU 上。要在不重新初始化任何张量的情况下正确做到这一点需要谨慎处理，且可能很繁琐。然而，FSDP2 允许在分片*之后*将张量实例化到 GPU 上（利用 `DTensor` 和 `nn.Module._apply` 方法的新 `swap_tensors` 路径）。