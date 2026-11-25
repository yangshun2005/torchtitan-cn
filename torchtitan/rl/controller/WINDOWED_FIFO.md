# 窗口化 FIFO 设计

本设计受 [MiniMax 论文](https://arxiv.org/pdf/2605.26494) 第 6.2.4 节所述的窗口化 FIFO 滚动调度启发。

## 符号

- `P`：每个训练步的提示组数（`num_prompts_per_train_step`）。
- `S`：目标稳态离策略步数（`target_offpolicy_steps`）。
- `B`：活跃缓冲区大小，以提示组计（`max_active_rollout_groups`），计算为 `B = (S + 1) * P`。
- `windowed_fifo_batches`：以训练批次计的 FIFO 前瞻窗口；`None` 表示无窗口。
- `W`：该窗口以组 ID 计的大小（`window_size`），计算为 `W = windowed_fifo_batches * P`。

## 贪心消费

当 `windowed_fifo_batches = None` 时，批处理器取缓冲区中任意位置最旧的已完成组。它不会因为一个未完成的较旧组而等待，即使有较新的已完成组就绪。

没有最大离策略年龄限制。慢组无论多旧，落地后即参与训练。超过 `S` 的样本占比通过 `train_batch/pct_samples_over_target_age` 上报，并以警告日志记录；这些样本永远不会被丢弃。

平均离策略年龄仍约为 `S`。根据 Little 定律，平均年龄 = 缓冲区中的组数 / 每步消费的组数：缓冲区容纳 `B = (S + 1) * P` 个组，其中 `P` 个是正在训练的批次，训练器每步消费 `P` 个，因此无论顺序如何，一个组大约等待 `S` 步。

## 掉队者问题

为什么要限制年龄？如果最旧的组很慢，贪心消费会持续训练较新的组，而慢组每步回来时都更旧。可选方案有：

- **停顿**：等待慢组，让训练器空闲。
- **丢弃**：训练较新的组，然后在慢组变得太旧时丢弃它。这浪费了已完成的滚动工作，并可能使进入训练的样本产生偏差。
- **提高目标离策略程度**：扩大活跃缓冲区，让每个组都能等待更久。这会让所有训练样本变旧，只为容纳少数掉队者。

窗口化 FIFO 是一种有界折中。它允许有限数量的较新组绕过慢组，为该组争取完成时间。被绕过的掉队者最多在比目标旧 `windowed_fifo_batches` 步时被消费，且不会为每个样本增加缓冲区大小。

窗口锚定在缓冲区中仍存在的最旧组上。消费较新的组不会使窗口前移，因此在批处理器等待最旧组之前，最多有 `W - 1` 个较新组可以绕过它。

## 窗口化 FIFO 配置

用户通过 `target_offpolicy_steps`、`num_prompts_per_train_step` 和 `windowed_fifo_batches` 配置 `S`、`P` 和 `windowed_fifo_batches`。`windowed_fifo_batches` 默认为 `None`。

控制器推导：

```text
B = (S + 1) * P
W = windowed_fifo_batches * P    (None when windowed_fifo_batches is None)
```

`windowed_fifo_batches = 1` 表示按批次 FIFO：最多 `P - 1` 个较新组 ID 可以越过卡住的头部。增大 `windowed_fifo_batches` 会让更多较新组暴露给调度器，而不增加 `B`。

```text
P = 8, S = 3 (B = 32)    W       max offpolicy steps
windowed_fifo_batches = None    none    unbounded
windowed_fifo_batches = 1       8       4
windowed_fifo_batches = 3       24      6
```

## 示例

设 `target_offpolicy_steps = 1`、`num_prompts_per_train_step = 3`、`windowed_fifo_batches = 1`。则 `max_active_rollout_groups = (1 + 1) * 3 = 6`，`window_size = 1 * 3 = 3`。组 `0`、`1`、`3`、`4`、`5` 很快完成，但组 `2` 很慢：

```text
[ 0 ready ][ 1 ready ][ 2 slow ][ 3 ready ][ 4 ready ][ 5 ready ]
```

批处理器消费组 `0` 和 `1`。窗口现在锚定在组 `2`，覆盖组 `[2, 4]`：

```text
anchored window
[ 2 slow ][ 3 ready ][ 4 ready ]   [ 5 ready ]
                                      ^ blocked outside the window
```

组 `3` 可以绕过组 `2` 并完成训练批次。组 `4` 也可以被消费，为组 `2` 争取更多完成时间。消费任一较新组都不会移动锚点，因此组 `5` 仍被阻塞，直到组 `2` 被消费。

当 `windowed_fifo_batches = None` 时没有窗口：组 `5` 也会被消费，组 `2` 在完成时即参与训练。

## 离策略上界

给定窗口大小 `W`，可以计算最坏情况下的离策略程度。考虑一个慢目标组 `g`，它在活跃缓冲区满时从尾部被接纳：

1. 当 `g` 被接纳时，它占据 `B` 个活跃槽位之一，因此满缓冲区中它前面最多有 `B - 1` 个较旧组。训练器先消费这些较旧组。
2. 因为 `B - 1 = S * P + (P - 1)`，这完成了 `S` 个训练步，并使下一批次差一个组。
3. 组 `g` 本可补全该批次，但它卡住了。
4. 训练器消费窗口内所有可能绕过 `g` 的 `W - 1` 个较新组。

```text
older groups                                  younger groups
[ 0 ][ 1 ] ... [ g-1 ][ g ][ g+1 ] ... [ g+W-1 ]
                       ^    at most W-1 prompt groups can bypass g
                       target group
```

```text
consumed_before = B - 1
consumed_after = W - 1
total_consumed = consumed_before + consumed_after = B + W - 2
```

每个训练步消费 `P` 个组，因此 `g` 被消费时的最大年龄为：

```text
max_offpolicy_steps = (B + W - 2) // P
```

代入 `B = (S + 1) * P` 和 `W = windowed_fifo_batches * P` 得 `max_offpolicy_steps = S + windowed_fifo_batches`。因此窗口化 FIFO 将最坏情况离策略程度增加 `windowed_fifo_batches` 步。无窗口时则无上界。