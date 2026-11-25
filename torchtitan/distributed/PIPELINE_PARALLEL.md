# 流水线并行

`pipeline_parallel.py` 的设计说明与调优参数。流水线并行把模型切分成多个阶段，放到不同 rank 上；本文说明 TorchTitan 模型如何编写以便于切分阶段、阶段如何获得一致的初始化，以及 FSDP 与发送调度如何与多阶段调度交互。

## 编写对流水线友好的模型
应用流水线并行时，你需要构造 nn.Module 对象，表示在给定流水线阶段上运行的那部分模型。无论你打算手动修改模型代码，还是用 tracing 之类的技术提取模型分块，对原始模型代码做几处改动就能大大简化这一过程。

### 简化顶层模型的 forward
大多数情况下，你可以让顶层 nn.Module 持有一串子模块并在 forward 中依次调用，把大部分复杂度交给子模块的 forward。如果你能把顶层 forward 简化为基本是对子模块调用的 for 循环，那么流水线切分任务就简化为：为每个阶段选择要保留的子模块集合。如果顶层 forward 中有不简单的逻辑，你就得想办法把这段逻辑补回到切分后的流水线阶段模型上，这会很麻烦。

示例（[PR #321](https://github.com/pytorch/torchtitan/pull/321)）：
我们过去在顶层 forward 中按 `seq_len` 切分 `freqs_cis` 缓冲区，再传入子模块，并期望子模块内的 `seq_len` 与其他本地张量的尺寸匹配。但在考虑 PP 切分时，我们并不知道是否应用了 TP，可能造成不匹配。在子模块内部用运行时准确的本地 `seq_len` 做 `freqs_cis` 切分同样简单，还能在 PP 切分时避开这个问题。

示例（[PR #322](https://github.com/pytorch/torchtitan/pull/322)）：
我们决定在每个 PP 阶段上复用顶层模型对象，只删除不需要的层，并确保顶层 forward 行为正确。这样就不必为每个阶段单独写一个把子模块拼起来的运行时 pp_forward。第一处改动是用 moduledict 而非 modulelist 存储层。这样即使删除部分层，层的完全限定名（FQN）也能保持不变——例如删除 layers.0 后 layers.1 仍是 layers.1，而用 list 则不然——这对检查点保存/加载很重要。保留 FQN 是使用分布式检查点（DCP）的前提，因为 DCP 用 FQN 作为分片元数据的全局唯一 ID。第二处改动是把输入层和输出层设为可选——层存在就运行，否则让输入直接透传。有了这两处改动，我们只需（meta）初始化整个模型，按阶段删除不用的部分，然后在加载检查点前把剩余部分物化到 GPU 上。

## 从种子检查点初始化各阶段
初始化流水线并行模型很有挑战，因为我们假设模型可能大到放不进本地 GPU（甚至可能放不进 CPU），同时我们希望使用与 1D 或 2D 并行模型（按位）相同的初始化，以便调试或比较不同运行。要改写原模型的 `init_weights` 函数，使其能只初始化部分层，并全局串行化初始化操作以获得一致的 RNG 顺序，并不容易。

目前我们用一种简单粗暴的方案绕开所有这些问题：在某个 CPU 实例上初始化整个模型，保存检查点文件，然后在阶段创建后借助分布式检查点的“加载”功能初始化该 PP 阶段上存在的 FQN。未来我们考虑为 `torch.pipelining` 增加更精细的初始化方案。

## 在流水线调度中控制 FSDP 驻留与 unshard 预取

循环流水线调度可能有多个由 FSDP 支撑的本地阶段。两个独立策略控制其参数生命周期：

- `pp_max_unsharded_active_stages` 限制可保持驻留的本地阶段数。降低该值可减少参数内存，但也可能引入额外的 reshard 和后续 unshard 周期。
- `pp_num_unshard_lookahead_factor` 控制多少个即将到来的不同阶段可以发起异步 all-gather。它只改变现有 `UNSHARD` 动作的发起位置；不改变阶段驻留、`RESHARD` 位置、集合通信次数，也不改变阶段首次使用其参数前的等待。

预取策略接受三种形式：

| 值 | PP rank `r` 上解析出的预取量 | 用途 |
| --- | --- | --- |
| `"full"` | `max_unsharded_stages` | 保持 PyTorch 的全窗口行为，在驻留窗口允许时尽早发起每个符合条件的 unshard。 |
| `"auto"` | `min(r + 2, max_unsharded_stages)` | 避免早期 PP rank 上出现全量 all-gather 突发，同时在有流水线预热窗口的 rank 上保留逐步增大的预取量。 |
| `(k0, ..., kN)` | `kr` | 针对非对称阶段开销、参数大小或网络行为的手动控制。 |

显式元组必须为每个 PP rank 提供一个正整数。每个值都不得大于解析出的 `pp_max_unsharded_active_stages`。预取量为 1 表示在该阶段之前立即发起 unshard；预取量为 2 表示当前阶段计算时保持一个未来阶段在途。

PyTorch 流水线调度默认 `"full"` 以保持向后兼容。TorchTitan 默认 `"auto"`，因为它掌握完整流水线拓扑，无需针对具体配方调优即可选择 rank 感知策略。复现原始 eager all-gather 发起模式时用 `"full"`；当性能分析确定了更好的 rank 特定距离时用显式元组：

```python
config.parallelism.pp_max_unsharded_active_stages = 4
config.parallelism.pp_num_unshard_lookahead_factor = "auto"

# 对于四 rank 流水线、驻留上限为四，等价于 "full"。
config.parallelism.pp_num_unshard_lookahead_factor = (4, 4, 4, 4)
```

## 限制待处理的流水线发送

`pipeline_parallel_max_outstanding_sends` 限制每个流水线 rank 上尚未到达等待点的前向和反向发送动作数。每个动作由方向、阶段和微批次标识，可能包含多个 P2P 操作。该限制不按微批次、张量、操作或字节计数。`None` 保持调度的正常因果等待，不添加硬性限制。

该选项仅适用于多阶段调度。PyTorch 将这些调度降为带显式 `WAIT_SEND_F` 和 `WAIT_SEND_B` 动作的全局通信动作序列。单阶段调度直接发起通信，不暴露该序列用于放置等待。

PyTorch 的流水线调度降级可能把已有等待提前以满足限制。这会提前释放调度对已发送输出的所有权，可能降低峰值内存。若匹配的接收尚未完成，较小的限制可能减少通信重叠。

降级过程会拒绝会造成依赖环的移动。例如，假设它想把发送 `S` 的等待移到本地动作 `A` 之前。该等待依赖对端接收 `R`。如果调度已把 `A` 排在 `R` 之前，该移动就会引入反向依赖：

```text
existing: A -> ... -> R
moved wait: R -> A
cycle: A -> ... -> R -> A
```

该过程能证明某个放置是否合法，但无法预测最快的限制值。等待可能因张量大小、拓扑、NCCL 时序和运行时争用而阻塞。内存收益也取决于张量大小和已发送存储的其他持有者。

从 `None` 开始。内存受限时，分析较小的值，并采用不降低吞吐量的最小值：

```python
config.parallelism.pipeline_parallel_max_outstanding_sends = 8
```