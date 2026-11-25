# 分布式 CPU KV 缓存（Mooncake Store）

Titan RL 使用 vLLM 的
[`MooncakeStoreConnector`](https://docs.vllm.ai/en/latest/features/mooncake_store_connector_usage/)，
通过 Mooncake Store（分布式 CPU DRAM 池）在各生成器之间共享 prompt 和解码 KV。若某一轮被路由到与上一轮不同的生成器（非粘性路由、过载后重新绑定），该轮从 Store 加载前缀，而非重新计算。若一个会话的每一轮都留在同一生成器上，本地 [SimpleCPU offload](SIMPLE_CPU_OFFLOAD.md) 更简单。

```text
turn 1 on generator A: prefill, save KV to the Store
turn 2 on generator B: look up prefix -> load KV from the Store -> prefill only new tokens
```

## 配置

在 controller 和生成器主机上安装 Mooncake SDK（`mooncake-transfer-engine`），并将每个生成器进程的 `MOONCAKE_CONFIG_PATH` 指向不含 `master_server_address` 的 Store 配置，例如：

```json
{
  "mode": "embedded",
  "metadata_server": "P2PHANDSHAKE",
  "global_segment_size": "16GB",
  "local_buffer_size": "64MB",
  "protocol": "rdma",
  "device_name": "mlx5_0",
  "enable_offload": false
}
```

`global_segment_size` 是每个 GPU 进程贡献的 CPU DRAM 大小。与 SimpleCPU offload 一样，需设置 `expandable_segments:False`（见
[配置](SIMPLE_CPU_OFFLOAD.md#setup)）。

controller 在其主机上启动作业的 `mooncake_master`，每个生成器进程将其地址填入 Store 配置副本，因此所有主机上的生成器共享同一个 Store。生成器通过 `MOONCAKE_REQUESTER_LOCAL_HOSTNAME`（默认：vLLM 的 `get_ip()`）读取彼此的 Store 内存；若生成器分布在多台主机上，需将其设为其他主机可达的地址，若为 IPv6 地址还需设置 `MC_USE_IPV6=1`。

## 生成器配置

```python
import uuid

from torchtitan.rl.generator.vllm.config import MooncakeStoreKVOffloadConfig
from torchtitan.rl.generator.vllm.engine import VLLMEngine

engine = VLLMEngine.Config(
    kv_offload=MooncakeStoreKVOffloadConfig(
        cache_prefix=f"my-run-{uuid.uuid4().hex}", save_decode_cache=True
    ),
)
```

- `cache_prefix` 为所有 Store 键划定命名空间。**每次启动都要使用新值**，包括从检查点恢复：Store 的生命周期长于一次运行，且策略版本会重启，旧命名空间可能返回由其他权重计算出的 KV。Store 键编码了 worker 的 TP rank，但不含 TP 度数、模型布局或 dtype；同一次运行的生成器共享同一引擎配置，因此这些一致。
- `save_decode_cache=True` 还会保存生成的 token，使下一轮复用上一次的补全结果。
- Titan 为每个生成器派生 vLLM 的 `lookup_rpc_port`（一个 IPC socket 名称）。

## 缓存键与权重更新

Store 键携带与 GPU 前缀缓存和 SimpleCPU offload 相同的 salt，权重更新语义也相同，包括热切换和混合权重 KV（见[权重与缓存生命周期](SIMPLE_CPU_OFFLOAD.md#weight-and-cache-lifecycle)）。由于 Store 是共享的，只有当两个生成器将组固定到相同的 `min_policy_version` 时，某一轮才会复用另一生成器的 KV。

## 多 GPU 协调

TP 或 DP > 1 的生成器使用 [SimpleCPU offload](SIMPLE_CPU_OFFLOAD.md#multi-gpu-coordination) 中的共享调度器。[Mooncake 适配器](mooncake.py)额外增加：TP leader 的 Store 查找会广播到其他 TP rank；本 TP 组的所有 worker 上报后保存才完成；每个 DP 副本启动自己的查找服务器。

## 与本地 CPU offload 的对比

两个连接器都不感知会话：键是前缀哈希，因此第 2 轮复用第 1 轮的块，是因为其 prompt 以第 1 轮的 prompt 和补全开头。会话处理由 Titan 负责：

| | 本地 CPU offload | Mooncake Store |
| --- | --- | --- |
| 跨生成器共享 | 否 | 是（同一 Master 上的生成器） |
| 粘性路由 | 是 | 是 |
| 轮次之间在 GPU 上保留会话前缀（`session_kv_holder`） | 是 | 是 |
| 优先驱逐已释放会话的 offload 块 | 是 | 否：生成器无法判断其他生成器是否仍需要某前缀，因此 Titan 从不删除 Store 键 |
| 保存的块 | 当 `lazy_offload=True` 时，仅保存 GPU 即将驱逐的块 | 每个已计算的块 |
| 驱逐 | CPU LRU 加会话释放提示 | Master 水位驱逐（近似 LRU） |

轮次之间（例如工具调用期间），会话的 Store 块会与所有新写入竞争，因此间隔较长时前缀可能已被驱逐。

## 性能

当轮次切换生成器、且从 Store 加载前缀比重新计算更快时，Mooncake 才有收益。在粘性路由（默认）下，很少有轮次切换生成器，因此本地 [CPU offload](SIMPLE_CPU_OFFLOAD.md) 能复用更多 KV。

### Agent RL 训练（粘性路由）

Terminal-Bench：Qwen3.5-27B，64K 上下文，粘性路由，2 台生成器主机，配置为 8 个生成器 x 1 GB300（TP=1，`gpu_memory_limit=0.30`）或 4 个生成器 x 2 GB300（TP=2，`gpu_memory_limit=0.21`）。Mooncake 走 RDMA（每主机 4 个 NIC），每 worker 40 GB，每主机一个 Master；本地 CPU offload 使用 `lazy_offload=True`，每 rank 40 GiB。每种 TP 的三个作业同时启动；取训练第 11-24 步的均值。每格显示 TP=1 / TP=2：

| 连接器 | KV 命中率，TP=1 / TP=2 | Prefill 毫秒，TP=1 / TP=2 | 词元间延迟毫秒，TP=1 / TP=2 |
| --- | ---: | ---: | ---: |
| 无 offload | 8.2% / 4.0% | 1,076 / 830 | 48.0 / 59.8 |
| Mooncake | 35.0% / 36.3% | 670 / 497 | 38.5 / 41.1 |
| 本地 CPU offload | 68.3% / 71.1% | 398 / 289 | 30.2 / 30.4 |

Mooncake 保存每个已计算的块，因此 Store 约半分钟就会轮换一遍，而非保留 GPU 即将驱逐的块。训练步时间差异在步间噪声范围内。

### 生成器微基准（每轮都在不同生成器上）

针对 Mooncake 目标场景的独立纯生成基准：无训练，会话的每一轮都被强制放到不同生成器上，因此本地 offload 永远无益。Qwen3-8B，单台 8x H100 主机上 4 个生成器，所有 Store 流量走一个 200 Gb/s NIC。每个会话从 8K 或 32K token 上下文开始，共 4 轮；会话 s 的第 t 轮在生成器 (s + t) % 4 上运行，增加 192 个输入和 32 个输出 token。运行 24 个会话（96 个请求）的耗时：

| 上下文 | 每生成器 TP | 无 offload（秒） | Mooncake（秒） | 加速比 |
| --- | --- | ---: | ---: | ---: |
| 32K | 1 | 41.3 | 30.7 | 1.35x |
| 8K | 1 | 11.3 | 10.6 | 1.07x |
| 32K | 2 | 24.8 | 30.9 | 0.80x（更慢） |

当重新计算上下文的成本高于加载其 KV 时，Mooncake 才有收益：在 32K 且 TP=1 时，一轮节省约 1.8 秒 prefill，而 Store 查找和加载增加约 0.85 秒排队。TP=2 时 prefill 足够快，通过共享 NIC 加载 32K token 的 KV 反而更慢。

## 验证

```bash
pytest -q tests/rl/unit_tests/cpu/test_mooncake_coordination.py
```

RL CI 镜像没有 Mooncake SDK，因此使用真实 Store 的端到端运行需手动进行。已用 `mooncake-transfer-engine==0.3.13.post1` 验证。