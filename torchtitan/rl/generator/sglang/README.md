# SGLang 生成器引擎

**状态：** vLLM 引擎的替代方案。仅在 SGLang `851642ade` 上验证过，`scripts/install_rl_env.sh --sglang` 安装的就是该版本；CI 在 B200 上运行 `test_sglang_bitwise_parity.py`（`integration_test_rl_b200.yaml`）。它依赖 SGLang 的私有 API（见链接到 sgl-project/sglang issue 的 TODO），因此其他 SGLang 版本可能无法使用。

`SGLangEngine` 在 [SGLang](https://github.com/sgl-project/sglang) 中运行生成器的 TorchTitan 模型，正如 `VLLMEngine` 在 vLLM 中运行一样。SGLang 提供 KV 缓存管理（分页 KV 池、基数前缀缓存）和调度（连续批处理、分块预填充、优先级、重叠调度、CUDA 图）；模型及其注意力内核来自 TorchTitan。

```
生成器引擎循环（每个 rank，SPMD）
  SGLangEngine
    add_requests ----> Scheduler.process_input_requests   （无 tokenizer/detokenizer 进程）
    step ------------> Scheduler.event_loop_overlap 的一次迭代
                          ModelRunner -> SGLangModelWrapper.forward(input_ids, positions, forward_batch)
                                           TorchTitan Decoder
                                             Attention -> SGLangInnerAttention -> RadixAttention
                                                            TorchTitanVarlenAttnBackend：
                                                            SGLang 的分页 KV 元数据，
                                                            torch varlen FlashAttention（训练器的内核）
                                             GDN / KDA -> SGLangInnerGDN / SGLangInnerKDA -> RadixLinearAttention
                                                            TorchTitanGDNAttnBackend / TorchTitanKDAAttnBackend：
                                                            SGLang 的状态槽，训练器的 Attention Gym 内核
                                           lm_head + TP gather -> SGLang LogitsProcessor / Sampler
    load_model_state_dict -> ModelStateDictLoader（FSDP 组，原地加载：CUDA 图保持有效）
```

每个 rank 在进程内构建自己的 SGLang `Scheduler`。SGLang 的 TP 调度器做出相同的调度决策，这是引擎循环已经依赖的约定。

## 使用方式

通过配置类型选择：

```python
from torchtitan.rl.generator.sglang.engine import SGLangEngine

generator=Generator.Config(..., engine=SGLangEngine.Config())
```

`torchtitan_recipes.rl.alphabet_sort.rl_grpo_qwen3_0_6b_varlen_sglang` 是一个完整配方（6 个 GPU：4 个生成器 + 2 个训练器）。

## 环境

SGLang 锁定的 PyTorch 版本比 TorchTitan 的 nightly 更旧，因此将其安装到 RL 环境之上的独立环境中，且不替换 PyTorch：

```bash
uv venv --python 3.12 .venv-sglang && source .venv-sglang/bin/activate
# RL 环境：PyTorch 和 vLLM nightly（batch-invariant 模式使用 vLLM 的
# batch-invariant bmm）、TorchStore 和 FlashAttention；然后安装已验证提交的
# SGLang，保留 RL 环境的 PyTorch。
scripts/install_rl_env.sh --sglang
```

## 支持的功能

| 功能 | 状态 |
| --- | --- |
| 稠密 softmax 注意力，varlen 后端（Qwen3、Llama 3） | 支持 |
| 张量并行 | 支持；在 batch-invariant 模式之外，模型的 TP all-reduce 运行在 SGLang 的 attention-TP all-reduce（自定义 all-reduce）上，而非 NCCL |
| CUDA 图（decode 和 prefill）、重叠调度、分块预填充 | 支持 |
| 按 rollout 组优先级调度 | 支持 |
| rollout 组内的 prompt KV 复用 | 支持；线性注意力（GDN、KDA）复用 prompt 最后 64 token 边界处的状态；见下文 |
| 模型 state dict 加载（`cpu_prefetch`、`gpu_fetch`） | 支持 |
| 策略版本前缀缓存隔离 | 支持（`cache_salt`） |
| `reset_kv_cache_on_model_state_dict_load` | 支持：运行中的请求被撤回并重新计算其 KV |
| 多轮会话 KV | 支持，始终开启；见下文 |
| CPU KV 卸载（SGLang HiCache） | 通过 `extra_server_args` 支持；见下文 |
| 跨生成器共享 KV（Mooncake Store） | 通过 `extra_server_args` 支持；见下文 |
| Batch-invariant 模式 | 支持；见下文 |
| 每请求指标（队列时间、TTFT、预填充、解码、抢占、缓存 token） | 支持 |
| 生成器内的数据并行 | 配合专家并行支持，作为 SGLang 的 attention DP；见下文 |
| 滑动窗口注意力（如 Muse Glimmer） | 支持；每个 token 的 KV 都保持缓存（无滑动窗口 KV 池） |
| GDN 线性注意力（Qwen3.5） | 支持；见下文 |
| KDA 线性注意力（Kimi K3） | 支持；见下文 |
| MLA（DeepSeek V3） | 支持，KV 缓存中按头存储 K/V，与 vLLM 引擎在 batch-invariant 模式下运行 MLA 的方式相同；不使用 SGLang 的压缩潜在 MLA 缓存 |
| Flex attention、attention sinks（GPT-OSS） | 不支持 |
| 在 DP 和 TP rank 上使用专家并行的 MoE、Dist-MoE | 支持；见下文 |
| 序列并行 | 不支持 |

不支持的配置在引擎构建时抛出异常。

## CUDA 图

`SGLangEngine.Config.cuda_graph` 决定 SGLang 捕获什么。SGLang 将 decode 和 prefill 放在不同批次中运行；`"decode"` 为 decode 批次捕获完整图。默认值 `"full"` 还会将 prefill 批次捕获为 SGLang 的完整 prefill CUDA 图，每个 token 数量桶一张，每个运行中的请求一个请求槽。没有这些图时，prefill 受限于 eager 模型的 CPU 开销（在 TP 2 下，Qwen3-0.6B 的 16K token prefill 块，CPU 开销约为 GPU 时间的 7 倍）。SGLang 的线性注意力后端无法捕获完整 prefill 图，因此线性注意力模型（GDN、KDA）在 `"full"` 下会抛出异常，需要 `"breakable"`：SGLang 的可中断 prefill 图，模型在图段中运行，中间注意力为 eager 模式，每个 prefill 步骤每层都有 CPU 开销。捕获会增加约 30 秒启动时间；`"decode"` 跳过捕获。

注意力内核在每种内核变体首次使用时编译，因此一次运行的首批 prefill 批次可能需要几秒钟。

## MoE

使用专家并行时，MoE 层将每一步的 token 分片到 TP rank 上，因此模型将一步填充到 TP 度的倍数；注意力跳过填充行。all-to-all token 分发器在主机上读取其拆分大小，CUDA 图无法捕获：需要 `cuda_graph="none"`。DeepEP 配合 `cuda_graph_compatible` 和 HybridEP 配合 `non_blocking_capacity_factor` 分发时无需主机同步，因此可在 CUDA 图下运行；没有这些选项时也需要 `cuda_graph="none"`。它们的 `num_max_tokens_per_rank` 必须覆盖每 EP rank 每步的 token 数，即 `(chunked_prefill_size + max_running_requests) / TP`，使用每 DP rank 的值；引擎在构建时检查。HybridEP 需要 EP 度可被 4 整除：一步可以有任意 token 数，否则 HybridEP 的 16 字节对齐会拒绝。DeepEP 的确定性分发（`DebugConfig.deterministic`）无法在 CUDA 图中运行，HybridEP 的 `non_blocking_capacity_factor` 必须为 1.0（低于此值 HybridEP 会丢弃 token）。`scripts/install_rl_env.sh` 不安装 DeepEP 或 HybridEP；需针对环境的 PyTorch 构建它们，如 `.github/scripts/install_deepep_v2.sh`（DeepEP）和 `.ci/docker/common/install_deepep.sh`（HybridEP）所做。SGLang 自带的 DeepEP wheel 在 PyTorch nightly 下无法通过 NCCL 版本检查。Dist-MoE（`Generator.Config.dist_moe_runtime`）在 CUDA 图下运行：decode 图，以及可中断或完整 prefill 图。即使没有专家并行，它也将一步的 token 分片到 TP rank 上，因此 TP > 1 时也需要专家并行。

## 数据并行

`data_parallel_degree` > 1 为专家并行提供 rank（Generator.Config 要求 `expert_parallel_degree == data_parallel_degree * tensor_parallel_degree`）。引擎将其作为 SGLang 的 attention DP 运行：一个 SGLang 实例覆盖 DP x TP 世界（`tp_size = DP * TP`，`attn_dp_size = DP`）。每个 DP rank 运行自己的请求，注意力头在其 attention-TP 组上分片，MoE 层在所有 rank 上运行 EP。对于 DP 2 x TP 2：

```
world = 4 个 rank，DP 2 x TP 2
SGLang TP 组（= world）：   [0 1 2 3]   <- MoE 层的 EP 4
attention-TP 组：         [0 1] [2 3] <- 头在这些组上分片
DP rank 0 = rank 0, 1       DP rank 1 = rank 2, 3
```

注意：SGLang 以其整个实例命名组。SGLang 的 "TP"（`tp_size`、`tp_group`、`tensor_model_parallel_all_reduce`）跨越所有 DP x TP rank，而其 "attention TP"（`attn_tp_size`、`attn_tp_group`、`attn_tp_all_reduce`）是一个 DP rank 的 TP rank：即 torchtitan 的 `tensor_parallel_degree`。因此引擎的 TP 代码（注意力头分片、模型的路由 TP all-reduce）使用 attention-TP 名称。两者在 DP = 1 时重合。

EP 集合通信跨越 DP rank，因此每个 DP rank 一起步进：SGLang 在空批次上运行没有请求的 DP rank，模型将其填充为每 TP rank 一行，且当任何 DP rank 有请求时，每个 rank 上的 `has_unfinished_requests` 都为 True。`max_num_seqs` 和 `max_num_batched_tokens` 是每 DP rank 的。

## GDN 和 KDA

SGLang 将每个请求的 GDN 或 KDA 卷积和循环状态保存在其状态池中，其基数缓存对它们做快照，因此会话的下一轮从缓存前缀恢复，与 KV 一样。训练器的 Attention Gym 内核推进状态（`TorchTitanGDNAttnBackend` 和 `TorchTitanKDAAttnBackend`，通过 SGLang 的线性注意力模型注册表注册）。它们使用 SGLang 的 `extra_buffer` mamba 缓存策略，保持重叠调度：SGLang 为每个请求提供两个跟踪槽，后端在基数缓存跟踪的位置将请求状态快照到其中一个槽，因此下一批次可以继续推进请求自己的槽。

- 预填充：在新 token 的最后 64 token 边界处。卷积历史从输入复制，Attention Gym 写入循环状态（`StateCheckpoints`）。
- 解码：SGLang 每 `mamba_track_interval` 个 token（64；SGLang 默认为 256）复制状态。

有缓存 prompt 的请求复用 prompt 最后 64 token 边界处的状态，会话的下一轮从上一轮的最后快照恢复，最多在其结束前 63 个 token。

在 batch-invariant 模式下，KDA 状态是最后一个完整 64 token 块之后的状态加上 Attention Gym 对未完成块的重放状态，与 vLLM 引擎相同。SGLang 的状态池将重放张量作为每个槽的额外卷积状态条目保存（`TorchTitanKDAConfig`），因此其基数缓存随状态一起对它们做快照；在头维度 128 时，它们占用循环状态 1.25 倍的内存。SGLang 从 extend 开始计算预填充快照位置，分块预填充可能使其偏离 64 token 网格，因此 KDA 预填充快照是其最后一个完整块处的状态（`StateCheckpoints`）加上该块之后的输入作为其重放状态（`copy_replay_tails`）。解码快照在 64 token 的倍数处，此时重放状态为空；SGLang 不复制重放条目，因此后端将跟踪槽的重放计数清零。

### KV 和状态共享一个池

对于 GDN 和 KDA 模型，引擎开启 SGLang 的统一内存：完整注意力层的 KV 缓存和 GDN / KDA 状态共享一个池，因此没有状态使用的内存可以存放 KV，反之亦然。运行中的请求占用 5 个状态槽（3 个，加上 `extra_buffer` 的 2 个跟踪槽），因此 SGLang 最多运行 `状态槽 // 5` 个请求（在 GB300 上，Qwen3.5-9B 在 TP 1、`gpu_memory_limit=0.86` 时约为 428）；如需更多，在 `extra_server_args` 中提高 `max_mamba_cache_size`。统一内存使约 25 ms 的解码步骤多花约 1 ms。

## 多轮会话

每个请求将其 rollout 的 `routing_session_id` 作为 SGLang 会话携带。当一轮结束时，SGLang 的基数缓存引用会话的缓存前缀（prompt 和 completion），直到 rollout 结束且引擎循环释放会话或其组。驱逐先取未引用的 KV，仅在没有其他可驱逐时才取引用的 KV，因此下一轮通常复用整个历史；没有请求会等待被持有的会话。

## Rollout 组

一个 rollout 组的样本共享一个 prompt。SGLang 仅在请求的预填充之后才将其 KV 插入基数缓存，因此如果一组的样本在同一批次中预填充，每个都会计算整个 prompt。（SGLang 的批内前缀缓存仅在 `lpm` 策略下重排等待队列，而优先级调度排除了该策略。）

因此引擎将具有给定 prompt 的第一个请求放入 SGLang 的等待队列，并将其他请求保持在队列外，直到该请求的预填充完成。被保持的请求随后从基数缓存读取 prompt 的 KV，仅计算 prompt 的最后一个 token（当页大于一个 token 时，计算其最后一个不完整页）：

```
add_requests([g7/s0, g7/s1, ..., g7/s15])   相同 prompt，相同 cache salt
  等待队列：g7/s0                      保持：g7/s1 .. g7/s15
step()  预填充 g7/s0 -> 其 prompt KV 进入基数缓存
  等待队列：g7/s1 .. g7/s15            每个仅预填充最后一个 token
```

对于 Qwen3-4B（TP 1）和同时准入的 8 个 prompt x 16 个样本，prompt token 的缓存份额从 0% 升至 94%（1K token prompt，TTFT p50 0.46 s -> 0.10 s），从 59% 升至 94%（4K token prompt，0.73 s -> 0.25 s）。被保持的请求保留其优先级和等待队列进入时间，因此组仍按最旧优先运行。线性注意力（GDN、KDA）模型也保持请求：被保持的请求从 leader 在 prompt 最后 64 token 边界处的状态快照恢复，并重新计算 prompt 的其余部分（见 [GDN 和 KDA](#gdn-and-kda)）。

## CPU KV 卸载

SGLang 的分层缓存将从 GPU 内存驱逐的 KV 保存在主机内存中，并在前缀命中时加载回来：

```python
SGLangEngine.Config(
    extra_server_args={
        "enable_hierarchical_cache": True,
        "hicache_size": 64,  # 每 rank 的主机内存 GB 数
        "hicache_write_policy": "write_back",
    }
)
```

加载回来的 KV 与重新计算的 KV 按位相同（在多轮工作负载的 batch-invariant 模式下检查过），模型 state dict 加载时的重置也会清除主机缓存。对于 GDN 模型，它还将被驱逐前缀的卷积和循环状态保存在主机内存中：前缀被加载回来的缓存轮次与训练器的预填充按位匹配。当重新计算前缀的成本高于复制其 KV 时，HiCache 才划算：对于 Qwen3-0.6B，重新计算更快。

KDA 模型目前与 HiCache 一起会抛出异常（#6015）。TODO：使用 HiCache 的 `write_through` 策略（SGLang 默认）时，KDA 一致性类中 18 次运行有 1 次与训练器不匹配（最大 logprob 差值 3.1e-2）；未找到根因。

## 跨生成器共享 KV（Mooncake Store）

[Mooncake Store](https://github.com/kvcache-ai/Mooncake) 在 HiCache 的主机内存之下增加一个 KV 层，所有生成器共享，因此会话的下一轮如果落在另一个生成器上，就加载其历史的 KV 而非重新计算：

```text
生成器 A 上的第 1 轮：预填充、解码；HiCache 将 KV 复制到主机内存和 Store
生成器 B 上的第 2 轮：GPU 和主机未命中，Store 命中 -> 加载历史的 KV，
                       仅预填充新 token
```

`scripts/install_r