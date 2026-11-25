# 模型配置转换

模型配置转换会重写完整的模型配置树。`build_model_config` 在转换运行前构建基础模型。

转换是构建和维护配置的一种受支持方式。它是可选的，用户也可以用自己的工具构建配方。

## 使用转换

先设置所有训练选项，然后调用一次 `apply_transforms`。

```python
config = muse_glimmer_30b()
config.parallelism.context_parallel_degree = 8

config = apply_transforms(
    config,
    [
        ContextParallelTransform(
            inner_attention_map={
                FlexInnerAttention: AllGatherKVFlexInnerAttention,
                SlidingWindowFlexInnerAttention: (
                    AllGatherKVSlidingWindowFlexInnerAttention
                ),
            }
        )
    ],
)
```

常见的注意力和前馈配置默认包含同步张量并行投影角色。要让这些集合通信与相邻的 GEMM 重叠，可用转换选择异步实现：

```python
config.parallelism.tensor_parallel_degree = 8
config = apply_transforms(
    config,
    [
        AsyncTensorParallelTransform(
            enable_sequence_parallel=config.parallelism.enable_sequence_parallel
        )
    ],
)
```

没有 TP mesh 时，同步投影类表现为普通线性模块。

异步 TP 转换精确匹配投影的所属类。它转换 `ColumnParallelLinear` 和 `RowParallelLinear`，但保留任意子类不变，因为用异步基类替换会丢弃其专门的 forward 行为。禁用序列并行时，构造异步 TP 转换会在任何转换发生前报错。

转换运行在已按其并行度分片的模型配置上（`build_model_config(..., parallelism=...)`）。替换配置的转换会保留其 `sharding_config`，或设置其替换实现所需的分片；它绝不会为整个模型重新推导分片。

`apply_transforms` 会深拷贝 trainer 配置。它先排序并应用转换，再校验结果，返回修改后的副本。若某个转换失败，输入配置保持不变。

在同一次 `apply_transforms` 调用中，量化和 `LMHeadFP32OutputTransform` 在 `LoRATransform` 之前运行。LoRA 会把每个目标投影配置替换为适配器，把其他所有配置替换为冻结子类。这些转换只转换标准投影配置，遇到冻结配置会报错，因此要与 LoRA 在同一次调用中应用，或在更早的调用中应用。

量化和 `LMHeadFP32OutputTransform` 在针对不同 Linear 时可以组合，例如 `MXFP8LinearTransform(fqns=["layers."])` 把 `lm_head` 留给 fp32 转换。同一次调用中量化先运行。lm_head 转换遇到已量化的 `lm_head` 会报错，量化遇到 `HiMidLoLinear` 会报错。

注意：量化后接 LoRA 时，LoRA 会冻结原始权重，但当前量化线性实现仍会在每次 forward 或 FSDP unshard 时重新生成量化权重操作数。对冻结权重而言这是可避免的工作。TODO：跨 forward 缓存其量化操作数。

同步张量并行边界可与量化和 LoRA 组合。它们的转换会替换投影计算，同时保留其列并行或行并行角色。异步张量并行尚不支持被转换的投影；它与 `LoRATransform`、`MXFP8LinearTransform` 和 `NVFP4LinearTransform` 冲突。分组量化只改变路由专家，而异步 TP 从不转换它们。

没有 trainer 配置时使用 `transform_model_config_`。它原地重写模型配置并返回根节点，不拷贝也不校验配置。

```python
model_config = build_model_config(
    "0.6B", attn_backend="varlen", parallelism=ParallelismConfig()
)
model_config = transform_model_config_(
    model_config, [LoRATransform(handlers=(LinearLoRAHandler(),))]
)
```

## 什么应放在这里

用 `build_model_config` 选择基础架构、注意力算法和注意力元数据格式。例如，FlexInnerAttention 消费 `BlockMask`，而 VarlenInnerAttention 消费累积序列偏移。

对于在已构建树中替换或包装配置的选项，使用转换。上下文并行、TP GEMM 后端、MoE 通信后端、量化和 LoRA 属于转换。量化模块、张量和内核位于 `torchtitan/quantization`。

CP 转换会为分布式执行特化所选注意力。它可以改变输入分片和预处理，但保留所选注意力算法和元数据格式。

图安全的 varlen K/V all-gather 支持连续分片和头尾分片。设置 `context_parallel_load_balancer=None` 可让每个 rank 只调用一次 varlen。头尾分片用两次调用平衡长因果文档。它也适用于仅滑动窗口模型，因为不需要注意力元数据。这些 varlen 后端不支持 PTRR。

## 依赖方向

本包可以导入其他 `torchtitan` 包。那些包不得导入本包。配方负责导入并应用转换。

共享类型放在本包之外。例如，`CPInnerAttention` 与注意力代码放在一起，只有安装它的转换属于本包。

## 编写转换

继承 `ModelConfigTransform`。转换选项使用仅关键字 dataclass。实现 `transform`，原地重写配置并返回模型根节点。仅在替换根节点时返回不同的配置。

```python
from dataclasses import dataclass

from torchtitan.config.transform import ModelConfigTransform, TransformRelations
from torchtitan.models.base.module import Module


class ExternalPrerequisiteTransform(ModelConfigTransform):
    def transform(self, model: Module.Config) -> Module.Config:
        return model


@dataclass(kw_only=True, slots=True)
class MyTransform(ModelConfigTransform):
    setting: int

    @classmethod
    def contribute_relations(cls, relations: TransformRelations) -> None:
        relations.add_precedence(
            before=ExternalPrerequisiteTransform,
            after=cls,
        )

    def transform(self, model: Module.Config) -> Module.Config:
        ...
        return model
```

转换只能看到模型配置。任何依赖模型运行方式的值都应作为转换选项传入，例如 `TokenDispatcherTransform` 的 `num_max_tokens_per_rank`，trainer 从 micro batch 推导它，推理引擎从每步 token 预算推导它。`apply_transforms(config, ...)` 适用于任何拥有 `model` 的配置，例如 trainer 或 RL 生成器配置。

用 `convert_config_type` 把一种配置实现替换为另一种。替换配置必须继承当前配置类型。这会保留先前转换的字段和包装器。

## 排序与冲突

可从 `relations.py` 安全导入的内置 TorchTitan 转换关系在那里声明。无法在那里导入的转换，例如急切导入可选依赖的转换或 TorchTitan 之外定义的下游转换，应重写 `contribute_relations`。用 `add_precedence(before=A, after=B)` 让 `A` 在 `B` 之前运行，用 `add_conflict(A, B)` 拒绝不兼容的一对。

每次应用都会根据中心关系和所选外部转换构建新图。关系作用于子类。冲突和排序检查在任何转换运行前完成，且只有所选转换诱导的图必须无环。无关转换保持输入顺序。

实例特定要求应使用配置校验，而不是关系。

## 校验

校验应放在哪里，见[配置校验](../README.md#validation)。`apply_transforms` 在最后一个转换后校验最终 trainer 配置。`ConfigLoader.load` 在启动选项后再次校验，`apply_overrides` 在注册的覆盖项后再次校验。

构造配置时也会运行 `__post_init__`。在调用 `apply_transforms` 前设置相关训练选项，它随后就能校验最终配置。

## TODO

许多转换和配置转换中使用的 `model.traverse` 逻辑，在模型配置于不同位置共享同一 `Config` 对象时表现不佳。