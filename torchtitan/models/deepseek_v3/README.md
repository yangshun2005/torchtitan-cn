## 下载 Tokenizer

```bash
# DeepSeek 671B tokenizer（自动下载 tokenizer.json 和 tokenizer_config.json）
python scripts/download_hf_assets.py --repo_id deepseek-ai/DeepSeek-V3.1-Base --assets tokenizer
```

```bash
# DeepSeek 16B tokenizer：
python scripts/download_hf_assets.py --repo_id deepseek-ai/deepseek-moe-16b-base --assets tokenizer
```

> **注意：** 我们复用 deepseek-ai/deepseek-moe-16b-base 的 tokenizer，方便用户测试和运行 16B 模型。这不是 DeepSeek-V3-16B 模型的官方 tokenizer。DeepSeek-V3 模型与 deepseek-moe 模型架构不同（注意力实现、MoE 路由实现等均不同），因此无法将 deepseek-moe-16b 模型权重加载到 DeepSeek-V3-16B 中。


## 训练

```bash
# 小模型快速调试运行
MODULE=torchtitan_recipes.tests.models.deepseek_v3 CONFIG=deepseek_v3_debugmodel ./run_train.sh
```

```bash
# 16B 参数模型：改编自 https://huggingface.co/deepseek-ai/deepseek-moe-16b-base 的旧版 16B 参数模型
MODULE=torchtitan_recipes.tests.models.deepseek_v3 CONFIG=deepseek_v3_16b ./run_train.sh
```

```bash
# 671B 参数模型
MODULE=torchtitan_recipes.models.deepseek_v3 CONFIG=deepseek_v3_671b ./run_train.sh
```

### 性能优化选项

为提升性能，DeepSeek-V3 模型可选用融合 Triton 内核，用于 MLA Q/KV 组装、ComplexRoPE 和 SwiGLU。这些覆盖项保持现有模型参数和检查点布局不变。

```bash
MODULE=torchtitan_recipes.models.deepseek_v3 CONFIG=deepseek_v3_671b ./run_train.sh \
  --override torchtitan_recipes.overrides.fused_mla.fused_mla \
  --override torchtitan_recipes.overrides.fused_swiglu.fused_swiglu
```

## HuggingFace -> DCP 检查点转换

我们实现了 StateDictAdapter，用于将 HuggingFace safetensor 转换为 DCP 格式。目前仅支持离线转换（使用 CPU 普通张量）从 HF 检查点转换为 DCP 检查点。

运行离线转换脚本：
```bash
python scripts/checkpoint_conversion/convert_from_hf.py <hf_checkpoints_dir> <dcp_output_dir> --model_name deepseek_v3 --model_flavor 671B
```