# 本地 CPU KV 缓存

Titan RL 使用 vLLM 的 [SimpleCPUOffloadConnector](https://github.com/vllm-project/vllm/blob/main/vllm/distributed/kv_transfer/kv_connector/v1/simple_cpu_offload_connector.py)，在 GPU 驱逐后从各 worker 的 CPU DRAM 重新加载 KV。独立生成器和 DP 副本各有独立缓存；多轮之间复用需使用粘性路由。

## 配置

经测试的 vLLM 版本会拒绝 KV 连接器的可扩展分配。导入 PyTorch 前设置：

```bash
export PYTORCH_ALLOC_CONF=expandable_segments:False
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
```

这两个作业级环境变量也会覆盖训练器的分配器默认值。禁用可扩展段会增加保留内存碎片，减少 DeepEP 缓冲区和 CUDA graph 池的可用余量。

## 生成器配置

```python
from torchtitan.rl.generator.vllm.config import SimpleCPUKVOffloadConfig
from torchtitan.rl.generator.vllm.engine import VLLMEngine

engine = VLLMEngine.Config(
    kv_offload=SimpleCPUKVOffloadConfig(cpu_bytes_per_rank=1 << 30),
)
```

容量按**每个 GPU worker** 计：TP=2 时此处为 2 GiB。需为权重缓冲区和其他主机进程留出空间。必须启用前缀缓存。

## 保存、驱逐与重载的内容

| 步骤 | 策略 |
| --- | --- |
| GPU -> CPU | 默认（`lazy_offload=True`）：仅保存 GPU 空闲队列头部附近（LRU 顺序）的缓存块，数量足以让约两步的 token 预算可被驱逐。`lazy_offload=False`（急切）：计算时即保存每个完整 prompt 块和解码块，包括单轮请求。 |
| CPU -> 丢弃 | 对 CPU 块池按 LRU 淘汰。启用 session holder 时，[会话释放](#会话释放)会将会话未使用的块移到队首，优先被驱逐。 |
| CPU -> GPU | 按需加载，无预取：调度时若请求的前缀查找命中 CPU，该请求等待其块加载完成。 |

## 权重与缓存生命周期

- CPU 块与 GPU 前缀缓存共享 salt，即请求的 `min_policy_version`：一个组在权重更新后复用其卸载的 KV，方式与复用 GPU KV 完全相同；新组以当前版本启动。salt 是组的起始版本，而非计算某块时的权重：热切换时，或组跨轮次固定于旧版本时，缓存的 GPU 和 CPU 块可能持有不同权重下计算的 KV。如需严格重算，使用 `reset_kv_cache_on_model_state_dict_load=True`，KV 卸载不支持该选项。
- 权重更新可热切换（`generator_router.hot_swap=True`）。保存和加载只复制 KV 块，权重加载不会触碰这些块；响应返回后，引擎继续推进待处理的传输。
- 保持 `generator.reset_kv_cache_on_model_state_dict_load=False`：重置无法释放仍在进行中的保存所持有的 GPU 块。

## 会话释放

通过 `session_kv_holder=VLLMSessionKVHolder.Config()` 启用 [session holder](session_kv_holder.py)，在多轮之间保留 GPU 前缀，并通过 `connector.release_session_cache(released_hashes, active_hashes)` 转发会话/组释放。`session_kv_holder=None` 时 CPU 卸载仍可用，但驱逐遵循连接器的常规 LRU 策略，没有会话释放提示。

```text
会话/组结束 -> 通知生成器 -> 等待请求和保存完成
            -> 优先驱逐未使用的 CPU 块
```

会话释放到达会话所固定生成器的 rank 0（粘性路由）；组释放到达每个生成器。Rank 0 将释放广播给其 worker。会话在重新固定后离开的生成器，会在组结束时释放其 KV。未启用 holder 时，生成器忽略释放。

已释放的块在被回收前仍可复用。本地 DP 副本上活跃会话使用的哈希和别名受保护。这是一次性提示；后续请求可再次缓存该前缀。仅 GPU 压力不会释放 CPU KV。

## 多 GPU 协调

每个 TP 调度器查询其本地 CPU 缓存。[SimpleCPU 适配器](simple_cpu_offload_connector.py)将保存完成限定在其 TP 组内；[共享调度器](kv_offload_scheduler.py)协调请求顺序与完成：

```text
本地查找 -> 分配 GPU 块 -> 各 worker 加载其 KV 分片
         -> 汇总 TP 完成 -> 恢复请求 / 释放引用
```

所有生成器 rank 同步是否仍有请求、调度器工作或排队输出。空闲 DP 副本用 dummy batch 加入必需的 EP 前向，直到所有工作排空。独立生成器不加入这些集合通信。

## 验证

完整矩阵需要八块可用 GPU。从仓库根目录运行：

```bash
pytest -q tests/rl/unit_tests/gpu/test_kv_offload.py
```

测试覆盖 CPU 恢复、会话释放、策略隔离、热切换、TP/DP/EP 和混合尾部，均不使用 CUDA graph。RL 多 GPU CI 作业运行 TP=2/DP=2/EP=4 和混合 TP=2 用例。FULL CUDA graph 已在 Qwen3-8B 上以 TP=2 和 TP=4 基准测试（CPU 重载与 GPU 前缀复用的 token 和 logprobs 一致），但未纳入 CI。不支持 `MultiConnector` 和自定义连接器。磁盘卸载尚未验证。