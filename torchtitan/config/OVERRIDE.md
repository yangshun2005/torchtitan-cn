# 可配置覆盖机制

通过导入注册覆盖的模块，即可将 torchtitan 训练配置中的任意 `Configurable`（模型组件、优化器、损失、数据加载器等）替换为替代实现，无需修改仓库内代码。本文档说明该机制的工作原理及如何编写覆盖。

## 动机

Torchtitan 采用配置驱动的构建系统：每个组件定义嵌套的 `Config` dataclass，由 `config.build()` 构造所属对象。目前替换组件需通过模型配置变换实现——遍历模型配置树，在模型构建前替换 `Config` 节点。

这种方式对仓库内的一等特性很有效，但*每种*替代实现都需要修改仓库。覆盖机制为三类工作消除了这一要求：

1. **硬件专用内核。** 针对特定硬件的高效 Triton/CUDA 内核，如旋转嵌入、融合注意力或自定义归一化。这些对性能有价值，但 (a) 硬件专用，(b) 更难维护，(c) 代码质量要求可合理低于核心。我们希望为厂商提供干净的扩展点，而非降低核心标准或直接拒绝。

2. **更大的融合区域。** 高效 MoE 可能将 dispatch–compute–combine 与自定义路由、通信和量化融合。这跨越多个 `Config` 节点，最好在更高作用域以单次替换表达（例如整体替换 `MoE.Config`）。

3. **团队实验。** 尝试非平凡实现而无需向核心渗透侵入性改动，之后可低成本保留或丢弃。

### 仓库内 vs. 外部

同一机制同时服务于仓库内示例（`torchtitan_recipes/overrides/`）和外部包，实现位置的选择对应上述目标：

- **仓库内（`torchtitan_recipes/overrides/`）**——说明性示例和广泛有用的内核，我们愿意以低于核心的标准维护。它们需显式启用，未请求时从不运行，因此不会导致默认行为回归。
- **外部包**——厂商或硬件专用代码，或任何我们不想在树内维护的内容。它位于厂商命名空间，通过列出其模块路径激活。

**非目标。** TorchTitan 中的算子覆盖 API。要替换或添加算子，请使用 PyTorch 的自定义算子路径，并将算子包装在 `Module` 中——参见[自定义内核与 `torch.compile`](#custom-kernels-and-torchcompile)。TorchTitan 不提供算子级覆盖接口。

## 设计概览

覆盖作者针对 `Configurable.Config` 类注册工厂。在配置构造之后、任何 `build()` 之前，机制遍历配置树，用工厂输出替换匹配节点。

### 核心原则

- **显式启用。** 除非用户在 `override.imports` 中列出目标，否则不执行任何操作。不从硬件或已安装包自动检测。
- **严格限定于请求。** 每个目标以 `module.function`（模块加 `@override` 工厂的函数名）精确指定一个覆盖，因此只应用指定的覆盖——绝不应用整个全局表。定义多个覆盖的模块需逐个列出函数来激活。
- **按实例定位，按节点冲突。** 覆盖通过 `fqns`（FQN glob）选择其认领的匹配节点。两个覆盖仅在认领*同一节点*（或一个认领另一个节点的祖先）时冲突——而非仅共享同一 Config 类。因此两个厂商在不同层上的归一化内核，或按层 A/B 测试内核，可干净组合；真正的冲突会报错。
- **任意 `Configurable`。** 覆盖作用于整个 `Trainer.Config` 树——模型组件、优化器、损失、数据加载器等——不仅限于模型。
- **最小接口。** 整个机制位于 `torchtitan/config/override.py`，复用量化与 LoRA 变换已使用的 `Configurable.Config.traverse()` + 替换模式。

## 工作原理

### 注册

覆盖模块定义替换实现，并用 `@override` 注册工厂：

下面的 `TritonRoPE` 是**说明性**示例（紧凑的子类添加一个字段的例子）；未提供 `triton_rope.py`。它是有效的覆盖：每个注意力模块拥有 `rope` 子模块（`RoPE.Config`），因此换入自定义 RoPE 是普通的组件覆盖。

示例针对 `ComplexRoPE.Config`，即其内核实现的格式。RoPE 格式读取 `positions` 的方式不同（文本用 1D id，视觉 RoPE 用 `(T, 2)` 坐标），因此针对你替换的格式；`target=RoPE.Config` 会匹配所有格式。视觉注意力同理：`VisionAttention.Config.rope` 持有 `ComplexVisionRoPE.Config` 或 `CosSinVisionRoPE.Config`。

```python
from dataclasses import dataclass

from torchtitan.config import derive, override
from torchtitan.models.common.rope import ComplexRoPE


class TritonRoPE(ComplexRoPE):
    """RoPE applied with a Triton kernel (illustrative)."""

    @dataclass(kw_only=True, slots=True)
    class Config(ComplexRoPE.Config):
        block_size: int = 128

    def __init__(self, config: Config): ...
    def forward(self, query, key, positions=None): ...


@override(target=ComplexRoPE.Config,
          description="Triton rotary embedding, ~2x faster on A100+")
def triton_rope(cfg: ComplexRoPE.Config, *, block_size: int = 128) -> TritonRoPE.Config:
    # `derive` copies every field shared with ComplexRoPE.Config; the factory states
    # only its deltas. See "Deriving the replacement config" below. `block_size`
    # defaults to 128 but can be tuned per config tree via a per-entry kwarg
    # (see "Optional per-entry kwargs" below).
    return derive(cfg, TritonRoPE.Config, block_size=block_size)
```

工厂接收匹配的配置作为第一个位置参数，返回其替换。它可声明关键字参数（如上面的 `block_size`），调用方通过 `(module_path, kwargs)` 导入条目填充；裸模块路径条目使用其默认值。工厂应用于*哪些*实例由 `fqns` 声明式控制（见[按实例定位](#per-instance-targeting)），而非在工厂内检查字段。

### 派生替换配置

优先使用 `derive(cfg, NewConfig, **deltas)` 而非手动复制每个字段。它通过复制与 `cfg` 同名的共享字段构建 `NewConfig`，再在其上应用 `deltas`；工厂只声明实际更改的内容。

这对版本健壮性很重要。手动复制会使工厂与其编写时目标的字段集耦合：

```python
# Fragile: ComplexRoPE.Config has many fields (theta, scaling, sharding_config,
# ...); hand-copying only some silently drops the rest to their defaults — and a
# field added later would be dropped too, with no error.
return TritonRoPE.Config(
    dim=cfg.dim, max_context_length=cfg.max_context_length,
                         theta=cfg.theta, scaling=cfg.scaling)

# Robust: every ComplexRoPE.Config field (including ones added later, inherited by the
# subclass TritonRoPE.Config) is copied automatically; only block_size is a delta.
return derive(cfg, TritonRoPE.Config, block_size=128)
```

语义：`cfg` 和 `NewConfig` 都有的字段从 `cfg` 复制；仅 `NewConfig` 有的字段来自 `deltas`（否则用其声明的默认值）；仅 `cfg` 有的字段被丢弃；delta 命名 `NewConfig` 上不存在的字段会报错。复制是浅层的（如 `dataclasses.replace`），这没问题，因为匹配节点正被替换并分离。当替换是目标的子类时（常见的直接替换场景），`derive` 最有价值——新的核心字段会被继承，从而免费携带过来。

**配置即代码。** 默认情况下，覆盖模块*就是*其配置：工厂内置合理默认值（上面 `block_size=128`），因此裸 `override.imports` 条目无需其他输入。

**可选的按条目 kwargs。** 当两个配置树需要*同一*覆盖以*不同*方式配置时，工厂可暴露关键字参数，每个 `override.imports` 条目可作为 `(target, kwargs)` 元组而非裸字符串提供；`kwargs` 转发给目标命名的覆盖。对于上面的 `triton_rope`，两个树可选择不同块大小而无需第二个模块：

```python
# one tree tunes block_size=128 (the default), another 256:
OverrideConfig(imports=["my_pkg.triton_rope.triton_rope"])                       # -> 128
OverrideConfig(imports=[("my_pkg.triton_rope.triton_rope", {"block_size": 256})])
```

工厂声明其接受的关键字参数（或 `**kwargs`）；它不接受的 kwarg 会报错而非静默无操作。在 CLI 上，将 kwargs 附加到目标为 `target=<json-object>`（作为单个 shell token 引用；`my_pkg.triton_rope.triton_rope` 是你自己覆盖的占位符）：

```bash
python -m torchtitan.train --module torchtitan_recipes.models.llama3 --config llama3_8b \
    --override 'my_pkg.triton_rope.triton_rope={"block_size": 256}'
```

JSON 保持值有类型（数字、`null`、字符串、嵌套），因此同一条目在 Python 或 CLI 中均可工作。我们仍避免在配置构造时解析的类型化覆盖 `Config`，那会重新将 recipe 耦合到覆盖包，违背无侵入目标。

### 激活

```bash
# Replace the torch-native SwiGLU activation with the Triton implementation:
python -m torchtitan.train --module torchtitan_recipes.models.llama3 --config llama3_8b \
    --override torchtitan_recipes.overrides.fused_swiglu.fused_swiglu

# Async tensor-parallel linear subclasses are preserved by the same override:
python -m torchtitan.train --module torchtitan_recipes.tests.models.llama3 --config llama3_debugmodel_dist_gemm \
    --override torchtitan_recipes.overrides.fused_swiglu.fused_swiglu

# A target with per-entry kwargs -- attached as target=<json>, quoted as one
# shell token (my_pkg.triton_rope.triton_rope is a placeholder for your override):
python -m torchtitan.train --module torchtitan_recipes.models.llama3 --config llama3_8b \
    --override 'my_pkg.triton_rope.triton_rope={"block_size": 256}'
```

### 应用

在 `Trainer.__init__` 中，任何组件构建之前：

1. 复制结构模型配置，并根据使用方的并行配置设置其分片。
2. 导入每个目标的模块——触发其 `@override` 装饰器。
3. 解析*活动*集：对每个 `module.function` 目标，取该函数注册的单个覆盖（按工厂定义模块及其名称匹配）。
4. 收集认领：遍历原始树，记录每个 `(override, node)` 对，按每个覆盖的 `fqns` 选择器过滤。
5. 检查按节点冲突（两个覆盖认领同一节点，或一个认领另一个节点的祖先）。
6. 应用：构建每个替换并换入。由于认领在任何变更前收集，应用与顺序无关（替换不会被重新遍历）。更改模块分片契约的覆盖必须设置替换的 `sharding_config`；契约不变时 `derive()` 会保留它。
7. 每个不同的 (override, 源类 -> 替换类, kwargs) 组记录一行，含匹配数和折叠的索引 FQN 范围（与 `format_config()` 一致）。`apply_overrides()` 每个被替换节点返回一行。

```
INFO: [Override] fused_swiglu: DenseFeedForward.Config -> DenseFeedForward.Config matched=32 fqns=['model.layers.{0..31}.feed_forward']
INFO: Applied 32 override(s)
```

模型配置是直接嵌套在训练器配置 `model` 处的 `Configurable.Config`。FQN 保留为从 `Trainer.Config` 根开始的完整路径，因此组件是 `model.layers.0.feed_forward`。RL `Controller.Config` 也持有 `override` 字段，在 actor 生成前应用于其整棵树；那里同一组件既是 `trainer.model.layers.0.feed_forward` 又是 `generator.model.layers.0.feed_forward`，因此覆盖会更改两个模型。RL 训练器与生成器之间必须不同的设置（例如 MoE token 调度器）应放在按角色的模型配置变换中。保留完整路径（而非重置为裸模型名）使按节点冲突检测能将整模型覆盖识别为组件覆盖的祖先。（这与量化变换的 `fqns` 不同，后者是裸的，因为变换直接遍历模型配置。）

### 外部包

```python
# vendor_x/overrides.py
from torchtitan.config import override
from torchtitan.models.common.moe import MoE


@override(target=MoE.Config,
          description="Vendor X fused MoE for XPU")
def vendor_x_moe(cfg: MoE.Config) -> "VendorXFusedMoE.Config":
    return VendorXFusedMoE.Config(num_experts=cfg.num_experts, ...)
```

```bash
pip install torchtitan-vendor-x
python -m torchtitan.train --module torchtitan_recipes.models.deepseek_v3 --config deepseek_v3_671b \
    --override vendor_x.overrides.vendor_x_moe
```

### 版本兼容性

覆盖依赖其读取的内部 `Config` 形状及其导入的模块，因此外部包应：

- **固定已验证的 torchtitan 版本范围。** 覆盖契约是目标的公共 `Configurable.Config` 字段；这些可能跨版本变化。
- **使其 pytorch-nightly 期望**与所固定的 torchtitan 版本**对齐**，因为硬件内核通常对 nightly 敏感。在包中记录测试过的 torchtitan × pytorch 组合。

## 按实例定位

定位是一等公民，而非事后考虑：许多动机场景按设计就会冲突（两个厂商的归一化内核，或按层 A/B 测试内核）。`@override` 上的 `fqns` 选择器声明式地说明覆盖认领*哪些*匹配节点——无需在工厂内嗅探字段。

`fqns` 为以下之一：

- `None`（默认）——`target` 的每个实例。
- FQN glob 列表——`fnmatch` 风格，其中 `*` 也跨越 `.`；节点 FQN 匹配任一 glob 即被认领。

默认情况下，`target` 匹配目标配置类及其子类的实例。若替换仅实现具体目标契约，向 `@override` 传 `exact=True`；子类配置则不被认领，因此不会记录为替换，仍可由子类专用覆盖处理。

FQN 是从 `Trainer.Config` 根开始的完整路径，例如模型组件是 `model.layers.0.feed_forward`，优化器是 `optimizer`。带 `*`（跨越 `.`）的 glob 保持选择器可读。

```python
# NVFP4 on every in-layer linear, leaving the top-level LM head ("...output")
# untouched (required for NVFP4; useful for MXFP8)
@override(target=Linear.Config,
          fqns=["*.layers.*.attention.*", "*.layers.*.feed_forward.*"])
def nvfp4_linear(cfg: Linear.Config) -> "NVFP4Linear.Config":
    return NVFP4Linear.Config(in_features=cfg.in_features, ...)

# Fused MoE only on the later layers
@override(target=MoE.Config, fqns=["*.layers.1[0-9].feed_forward"])
def vendor_moe(cfg: MoE.Config) -> "VendorMoE.Config":
    return VendorMoE.Config(num_experts=cfg.num_experts, ...)
```

`fqns` 仅接受 glob。通用谓词选择器（`(fqn, cfg) -> bool`）还可支持基于字段/类型的选择（例如"跳过已量化的 linear"），列于[未来工作](#future-work)。

### 按节点冲突检测

冲突按*节点*检测，而非按*类*。两个覆盖可自由针对同一 Config 类，只要其认领节点不相交——因此一个归一化内核在 `*.layers.0.*`、另一个在 `*.layers.1.*` 可共存，跨层 A/B 测试内核也可行。仅当两个不同覆盖认领**同一节点**，或一个认领另一个节点的**祖先**（嵌套替换，会依赖顺序）时报错。两种情况下的修复都是收窄 `fqns` 选择器使认领不相交。

认领在任何变更前从**原始**树收集，替换从不重新遍历，因此应用与顺序无关——一个覆盖绝不会静默影响另一个的匹配。

#### 父类与子类覆盖

同一规则直接回答父/子问题。假设覆盖 A 针对父 Config（如 `MoE.Config`），覆盖 B 针对嵌套其中的 Config（如 `GroupedLinear.Config`）：

- **不相交子树**（A 在 `...layers.0.feed_forward`，B 在 `...layers.1.feed_forward.routed_experts.w13`）——认领节点无关，两者都应用。
- **重叠**（A 在 `...layers.0.feed_forward`，B 在 `...layers.0.feed_forward.routed_experts.w13`）——B 的节点在 A 内，即上述祖先情况，因此**报错**。

我们报错而非隐式选择两种合理行为之一：

1. *先应用父类，再在新父类中覆盖子类。* 仅当 A 的替换仍暴露匹配子类时有意义，且会重新引入顺序（父先于子）和重新遍历——我们刻意避免。
2. *静默丢弃子覆盖*（当 A 的替换结构不同、无此子类时）。静默无操作正是隐藏 bug 的那类意外。

由于认领在原始树上收集，检查时甚至无法判断给定对会落入哪种情况——因此不