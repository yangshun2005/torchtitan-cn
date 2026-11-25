# 模型状态字典传输

`ModelStateDictTransport`（`model_state_dict_transport.py`）通过 `torch.distributed._transport` 后端（`backend="nixl"`，默认值，或 `"mooncake"`）的单边 RDMA 读，把训练器的模型状态字典搬入生成器的预取模型状态字典。它是默认的 `Controller.Config.model_state_dict_transport`；用以下方式选择 Mooncake：

```python
config.model_state_dict_transport = ModelStateDictTransport.Config(backend="mooncake")
```

它是一条独立路径，而非 TorchStore 传输：训练器和生成器的分片在初始化时已知，因此它一次性规划所有读，并直接暂存到已注册内存，无需 StorageVolume、键查找或每步元数据交换。训练器暂存一份已按生成器 rank 重新分片的固定 CPU 副本，因此优化器可在生成器读取时继续步进，且训练器不保留持久 GPU 副本。

`None` 则改用 TorchStore，`model_state_dict_load_mode="gpu_fetch"` 仍需要它。TODO：待本路径覆盖 `gpu_fetch` 后删除 TorchStore 路径。

Mooncake 需要 `mooncake-transfer-engine-cuda13==0.3.14`（由 `scripts/install_rl_env.sh` 安装）以及 2026-11-10 或之后的 PyTorch nightly（https://github.com/pytorch/pytorch/pull/203623；更旧的版本可能丢弃已完成的读并超时）。`scripts/install_rl_env.sh` 固定的 vLLM wheel 固定了 torch 2026-11-07，太旧。TODO：待该固定版本越过 2026-11-10 后删除此说明。没有 https://github.com/pytorch/pytorch/pull/204113 时，其他 NUMA 节点上的主机内存会使用节点 0 的 NIC，可能使读带宽下降数倍。在仅 IPv6 的主机上设置 `MC_USE_IPV6=1`；否则 Mooncake 的握手会通过 IPv4 解析对端主机名。没有 RDMA 设备时，Mooncake 回退到其 NVLink 传输，而后者无法读取固定 CPU 内存，读会失败并报 `Mooncake transfer submission failed`；请设置 `MC_FORCE_TCP=1`。

NIXL 需要 `nixl==1.5.0` 和 `nixl-cu13==1.5.0`（由 `scripts/install_rl_env.sh` 安装），并使用其 UCX 插件；用常见的 UCX 变量选择 NIC 和传输，例如 `UCX_NET_DEVICES`、`UCX_TLS` 和 `UCX_MAX_RMA_RAILS`。

## 布局

每个生成器具有相同的分片，因此每个训练器 rank 暂存其分片的一份副本，按生成器 rank（g0、g1）布局。一个分片块是训练器 rank 的张量盒与生成器 rank 的盒的交集，并转换为生成器的 dtype。生成器 rank 从每个训练器 rank 直接读入其已注册的预取缓冲区。

示例：一个形状为 (8, 6) 的张量，训练器 FSDP 4 在维度 0 上，生成器 TP 2 在维度 1 上。

```
        trainer staging buffers (pinned, registered)
        rank 0        rank 1        rank 2        rank 3
       +------+------+------+------+------+------+------+------+
       | g0   | g1   | g0   | g1   | g0   | g1   | g0   | g1   |
       | 2x3  | 2x3  | 2x3  | 2x3  | 2x3  | 2x3  | 2x3  | 2x3  |
       +------+------+------+------+------+------+------+------+
          |      |      |      |      |      |      |      |
          v      |      v      |      v      |      v      |   read
       generator rank 0 (g0): (8, 3), rows 0-1, 2-3, 4-5, 6-7
                 v             v             v             v
       generator rank 1 (g1): (8, 3), rows 0-1, 2-3, 4-5, 6-7
```

每个分片块是一次读，读入该生成器 rank 对应键的预取缓冲区，按分片块偏移顺序连续排列。如果某个键的所有分片块都是生成器分片的一段连续区间（如上例），则正好落在其应在位置。否则该键的预取缓冲区按打包方式存放其分片块，加载时再复制到位：例如，一个行并行权重（FSDP `Shard(0)`、TP `Shard(1)`）被按维度 0 切分的生成器 rank 读取时，到达的分片块只覆盖每行的一部分。这样每个分片块一次读，无需逐行读，也无需训练器集合通信，代价是加载时进行一次跨步复制。当多个训练器 rank 暂存同一分片块时（HSDP 副本），生成器 rank 以轮询方式分散到它们之上。如果训练器未覆盖生成器分片的每个元素，或遇到 `Partial` 及不均匀的 `_StridedShard` 放置，初始化会报错。

## 生命周期

1. 初始化（`ActorSystem.initialize_model_state_dict_transport`）：每个生成器报告其各 rank 的 `shard_layout`，它们必须一致（`shared_rank_layouts`）；每个训练器 rank 分配并注册其暂存缓冲区（`register_trainer_source`）；每个生成器 rank 注册其预取缓冲区，连接到每个训练器 rank 并规划其读（`connect_generator`）。
2. 推送：训练器转换其分片并复制到暂存缓冲区（`stage`），无集合通信。
3. 拉取：每个生成器 rank 发起其读（`read`）并等待，超过 `read_timeout_s`（600 秒）后报错。超时会使读仍处于进行中；请重启作业。
4. 加载：生成器从预取缓冲区复制其权重，解包打包的键（`copy_prefetched_`）。

控制器在推送 N+1 之前完成拉取 N，因此单个暂存缓冲区就足够。