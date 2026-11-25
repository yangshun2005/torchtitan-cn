# torchtitan 中的 FLUX 模型

## 概述
本目录包含 [FLUX](https://github.com/black-forest-labs/flux/tree/main) 模型在 torchtitan 中的实现。在 torchtitan 中，我们展示 FLUX 模型文本到图像部分的预训练流程。

## 前置条件
安装所需依赖：
```bash
pip install -r .ci/docker/requirements-flux.txt
```

## 使用方法
首先，使用你自己的访问令牌从 HuggingFace 下载自编码器模型：
```bash
python scripts/download_hf_assets.py --repo_id black-forest-labs/FLUX.1-dev --additional_patterns ae.safetensors --hf_token <your_access_token>
```

该步骤会从 HuggingFace 下载自编码器模型，并保存到 `assets/hf/FLUX.1-dev/ae.safetensors` 文件。

运行以下命令在单 GPU 上训练调试模型：
```bash
MODULE=torchtitan_recipes.tests.models.flux CONFIG=flux_debugmodel ./run_train.sh
```

如需使用其他配置训练，运行以下命令：
```bash
MODULE=torchtitan_recipes.models.flux CONFIG=flux_schnell ./run_train.sh
```

要从训练好的检查点生成图像，使用相同的启动器启动推理入口。提示词从 `config.inference.prompts_path` 读取（默认：[`inference/prompts.txt`](inference/prompts.txt)）：
```bash
ENTRYPOINT=torchtitan.models.flux.inference.infer \
MODULE=torchtitan_recipes.tests.models.flux CONFIG=flux_debugmodel_inference \
./run_train.sh
```


## 支持的特性
- 并行：模型支持 FSDP、HSDP、CP，可在多 GPU 上训练。
- 激活检查点：模型使用激活检查点降低训练时的内存占用。
- `torch.compile`：独立的损失函数编译。见下文 [torch.compile](#torchcompile)。
- MXFP8 量化：在 SM100+（Blackwell）硬件上对线性层进行动态 MXFP8 量化。见下文 [MXFP8 量化](#mxfp8-quantization)。
- 分布式检查点保存与加载。
    - 当前检查点实现的说明：为使模型权重的分片方式与检查点一致，需要在保存检查点前对模型权重分片。具体做法是在评估结束时检查每个模块，若为 FSDPModule 则对其权重分片。
- FLUX 模型的 CI。支持在 8 GPU 上定期运行集成测试，以及单元测试。

## torch.compile

损失函数可独立编译：

```python
config.model.local_compile.regions = ["loss"]
```

模型本身不由标准 Trainer 编译。

## MXFP8 量化

Flux 模型支持 MXFP8（Microscaling FP8）量化，用于在 SM100+ 硬件（B200、B100）上加速训练。该功能使用 torchtitan 现有的 `MXFP8LinearTransform`，将线性层动态量化为 MXFP8 精度。

**要求：**
- SM100+ GPU（如 NVIDIA B200、B100）
- `torchao` nightly

### 使用配置预设

预配置的 MXFP8 预设：

```bash
# Flux schnell with MXFP8
MODULE=torchtitan_recipes.models.flux CONFIG=flux_schnell_mxfp8 ./run_train.sh

# Flux dev with MXFP8
MODULE=torchtitan_recipes.models.flux CONFIG=flux_dev_mxfp8 ./run_train.sh
```

### 自定义配置

要创建自定义 MXFP8 配置，在 recipe 模块中定义新函数。`apply_transforms` 返回带有量化模型配置的配置副本：

```python
from torchtitan.config.transform import apply_transforms, MXFP8LinearTransform
from torchtitan.models.flux.trainer import FluxTrainer
from torchtitan_recipes.models.flux import flux_dev, flux_schnell


def my_custom_mxfp8() -> FluxTrainer.Config:
    config = flux_schnell()  # or flux_dev()
    return apply_transforms(
        config,
        [
            MXFP8LinearTransform(
                fqns=[
                    "double_blocks",
                    "single_blocks",
                    "img_in",
                    "txt_in",
                    "time_in",
                    "vector_in",
                    "final_layer",
                ],
            ),
        ],
    )
```

`fqns` 参数指定要量化的完全限定模块名。所列名称覆盖了 Flux transformer 中所有包含线性层的子模块。MXFP8 要求维度按 32 字节对齐；Flux 的默认隐藏维度满足该要求。

## TODO
- [ ] 更多并行支持（张量并行、流水线并行等）