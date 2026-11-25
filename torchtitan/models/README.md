本说明介绍在 `torchtitan` 仓库中添加新模型的流程。多数情况下，新模型应先添加到 `torchtitan/experiments` 文件夹下。贡献标准请参阅其中的[贡献指南](/torchtitan/experiments/README.md)。总体而言，请遵循 `torchtitan` 的[指导原则](/README.md#overview)。

离线探索也建议采用相同步骤，除非另有说明。

MoE 模型请参阅[专家并行后端选择指南](common/moe/README.md)。

## 添加模型

请参考 [Llama 3 文件夹](llama3) 作为示例。

文件夹结构应如下：
- `model.py`
  - 注意：请遵循指导原则，编写单设备模型代码。
  - 注意：我们优先考虑可读性而非灵活性。首选风格是不在不同模型间共享模块，最常见和最复杂的模块除外。
  - 定义一个继承自基础模型的 Model 类（例如 `torchtitan/models/common/decoder.py` 中的 `Decoder`）。
  - 模型类应包含一个嵌套的 `Config` dataclass（继承自基础模型的 `Config`），用于保存所有架构超参数。
    - `get_parameter_counts()` 报告模型总规模和活跃规模，`build_flops_estimator()` 估算每个原始 CPU 微批次的工作量。`Decoder` 子类继承这两个方法；新的 `BaseModel` 系列需自行实现。
    - `Decoder.Config.layers` 的类型为 `list[TransformerBlock.Config]`。如果 flavor 构建器或模型代码使用模型自身的 block 或 attention 配置类型，请用该类型重新声明 `layers`（以及 block 的 `attention`），并加上 `# pyrefly: ignore [bad-override]`，而不是在每次读取时用 `isinstance` 收窄类型。
  - 带模态编码器的模型有两个类：纯文本模型（`Decoder` 子类，例如 `Qwen35Model`），以及它和 `MultimodalModel` 的多模态子类（例如 `Qwen35MultimodalModel`），后者添加编码器，其 `Config` 要求这些编码器。在模型和 `Config` 的基类中都将文本类列在前面：dataclass 字段按反向 MRO 解析，否则 `MultimodalModel.Config` 中 `Decoder.Config` 字段的副本会覆盖文本 `Config` 的默认值。（`KimiK25Model` 仍将 `MultimodalModel` 列在前面；参见 `kimi_k2_7/model.py` 中的 TODO。）
  - `__init__()` 使用 `Config` 构建模型。
  - 参数初始化由每个模块 `Config` 上的 `param_init` 系统处理。在模型 flavor 构建器的每个子配置上设置 `param_init`（一个 `dict[str, Callable]`，将参数名映射到初始化函数）。`init_states()` 会自动递归到所有子模块，无需手动递归调用。需要设备感知的 buffer 初始化（例如 RoPE、MoE）时，重写 `_init_self_buffers()`。
  - 将数值需要主精度的参数（例如 HF checkpoint 以 float32 存储的参数）列在其模块 `Config` 的 `preserve_dtype_params` 中；FSDP 混合精度、checkpoint 导出和 RL 权重同步不会降低它们的精度。如果组件总是需要该精度，则在组件的 `Config` 上默认设置（例如 GDN 的 `A_log`）；如果只是通用模块的某个实例需要，则在 flavor 构建器中设置（例如 Qwen3.5 的 GDN `GatedRMSNorm`）。
  - 如果 `model.py` 变得过大或过于复杂，可添加额外文件以降低复杂度，例如用 moe.py 存放 `MoE`、router、grouped-linear 和专家激活模块。
- `state_dict_adapter.py`
  - 继承 [`BaseStateDictAdapter`](/torchtitan/models/base/state_dict_adapter.py)，实现 `torchtitan` 模型定义与其他模型定义（例如 HuggingFace 的定义，以便以 HF 格式保存/加载模型 checkpoint）之间的 state dict 映射。
  - 此类适配器有多种用途：
    - `scripts/checkpoint_conversion/` 中的 checkpoint 转换脚本会用它适配 CPU 上包含非分片 `torch.Tensor` 的 state dict。
    - 训练期间，[`CheckpointManager`](/torchtitan/components/checkpointer/dcp.py) 会用它适配 GPU 上包含（可能分片的）`DTensor` 的 state dict，以 HF 格式保存/加载 checkpoint。
    - 后训练中，`to_hf()` 可将 torchtitan 模型转换为 HF 模型，供其他框架推理使用。
  - 离线探索时此项可选。
- `sharding.py`
  - 定义声明式构造函数，返回模型自身组件的 `ShardingConfig`（或 `SpmdType`）。TP、SP、EP 和 inner-attention 局部 SPMD 区域以此方式表达，而非运行时 `parallelize_module` 计划。
  - 尽可能复用 `torchtitan/models/common/decoder_sharding.py`、`moe/sharding.py` 和 `vision_encoder_sharding.py` 中的共享构造函数（`token_embedding_sharding_config`、`norm_sharding_config`、`residual_activation_placement`、`colwise_sharding_config`、`rowwise_sharding_config`、`moe_sharding_config` 等）；若它们已覆盖整个模型，则跳过此文件。
  - 按规范的外层到内层 SPMD 顺序声明 mesh 轴：稠密部分（attention/MLP/norm/embed/lm_head）为 `(dp, cp, tp)`，稀疏部分（MoE 专家权重）为 `(dp_replicate, edp_shard, ep)`。`Module._parallelize` 根据声明的轴解析 mesh。
- `model.py`
  - `BaseModel.parallelize()` 依次应用声明式模型并行、激活 checkpointing 和 FSDP/HSDP。
  - 仅当模型需要不同的生命周期顺序时才重写 `parallelize()`；仅当模型有特定于模型族的 FSDP 结构时才重写 `_apply_fsdp()`。
  - 用 `state_dict_adapter_cls` 将 state dict 适配器绑定到模型类。
  - 语言模型的 CP 通过 `Decoder._cp_shard` 进行，它调用每个 CP attention 后端的 `prepare_cp_metadata`，然后对声明的张量输入调用 `context_parallel.shard_tensors`。
- `pipeline.py`（模型较小时可选）
  - 应用 PP
- `flavors.py`
  - 定义 `MODEL_FLAVORS`，将 flavor 名称映射到其配置构建器和能力上下文长度。
  - 定义 `build_model_config(flavor, *, parallelism=...)`，返回具体的、分片的 `Model.Config`。
  - 在各组件配置构建处，根据构建该配置所用的 `parallelism` 设置 `sharding_config`（序列并行和专家并行会改变声明的布局）。`torchtitan/models/common/config_utils.py` 中的 `make_*` 工厂接受 `enable_sp`/`enable_ep`，并对其构建的内容进行分片。Mesh 度数在运行时解析，因此未启用的轴会被跳过。
  - 使用不同并行的消费方（例如 RL 生成器）会构建自己的模型配置。
- `__init__.py`
  - 重新导出模型类、`MODEL_FLAVORS` 和 `build_model_config`，作为模型包的公共 API。
  - 仅当模型需要特定行为时才重写其 pipeline 或优化器 hook 方法。
  - 模型可重写的生命周期 hook 参见 [`BaseModel`](base/model.py)。
- 训练配方应放在 `torchtitan_recipes` 中，位于模型包之外。仅用于测试的配置放在 `torchtitan_recipes.tests` 中。
- `README.md`
  - 包含下载 tokenizer / 编码器的[说明](/README.md#downloading-a-tokenizer)。
  - 包含下载模型 checkpoint 以进行继续预训练或后训练的说明。
  - 更新当前开发状态，包括已支持的功能和即将推出的功能。
  - 离线探索时此项可选。

## 测试与基准测试
- 数值测试
  - 一种端到端方法是：将同一模型 checkpoint 加载到 `torchtitan` 模型和 HF 模型中，比较相同输入下的模型输出。这假设：
    - HF 实现正确。
    - `torchtitan` 模型和对应 state dict 适配器均正确，则两者都正确。
- 损失收敛
  - 若有已验证的基线，将损失曲线与基线比较。
  - `torchtitan` 内部的比较请参阅[指南](/docs/debugging.md#validating-a-distributed-training-technique)。
- 性能基准测试
  - 请参阅[基准测试](/docs/benchmarks/)文件夹。
- CI 测试
  - 包括单元测试和集成测试，参见[示例](/tests/)。
  - 如果模型文件夹位于 experiments 文件夹下，将测试放在模型文件夹下。否则，将测试放在 `/tests` 文件夹下。
  - 添加必要的 GitHub [workflow](/.github/workflows/)。

### FLOPs 估算

`Config.build_flops_estimator(model, seq_len=...)` 返回一个可调用对象，输入原始 CPU 微批次字典，输出分片前模型整体的逻辑训练 FLOPs。应基于完整模型构建；运行时估算应使用 CPU 元数据和张量形状，不进行设备到主机的同步，也不保留模型模块。原生 decoder 使用 `Attention.Config.flops(positions)`。多模态模型通过 `multimodal_encoder_fqns` 共享编码器组合。Muse 重写估算器，使用编码器持有的输出 token 数来添加输出模块的工作量。`ForwardBackwardResult.num_flops` 报告该 DP rank 原始微批次的未归约工作量，覆盖完整逻辑模型，并在 CP rank 间复制。训练器在 DP mesh 上对其取平均；CP rank 持有相同的值。

`Config.get_parameter_counts(model)` 报告唯一的参数总数和架构上活跃的参数数，包括嵌入和编码器，并按活跃比例对路由专家加权。FLOPs 单独排除仅用于查找的嵌入。每个 decoder attention 配置根据非填充查询在其文档中的位置计算开销：位置 p 的因果查询关注 p + 1 个 key，滑动窗口 attention 至多关注窗口大小，而线性 attention 对每个查询开销相同。填充 token 被排除。重计算和逐元素操作不计入。