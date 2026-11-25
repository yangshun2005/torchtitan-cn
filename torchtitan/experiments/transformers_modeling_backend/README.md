# Huggingface Transformers 建模后端

该后端支持用 4D 并行训练 HF transformers 模型。

本后端使用 TorchTitan SPMD 类型系统。

## 快速开始

- 依赖 `transformers==5.9.0`

- 测试配方：`torchtitan_recipes/tests/transformers_modeling_backend.py`
```diff
...
- --module torchtitan_recipes.tests.models.llama3
+ --module torchtitan_recipes.tests.transformers_modeling_backend
 --config transformers_modeling_backend_debugmodel
...
```
- 训练：`LOG_RANK=7 MODULE=torchtitan_recipes.tests.transformers_modeling_backend CONFIG=transformers_modeling_backend_debugmodel_compile ./run_train.sh`
    - 请确保已提前创建 tokenizer
<img width="1334" height="453" alt="image" src="https://github.com/user-attachments/assets/da459448-027b-4af9-8176-6a3e433a272c" />

## 支持的功能

- 以下模型已测试。这是后端覆盖范围，不是推荐的生产训练配方：
    - Dense（FSDP/CP/TP/PP/`torch.compile`）
        - `meta-llama/Llama-3.2-1B`
        - `microsoft/phi-2`
        - `Qwen/Qwen2.5-7B`
        - `mistralai/Mistral-7B-v0.1`
        - `ByteDance-Seed/Seed-Coder-8B-Instruct`
        - `Qwen/Qwen3-4B-Instruct-2507`
        - `arcee-ai/AFM-4.5B`
        - `ibm-granite/granite-3b-code-base-2k`
        - `baidu/ERNIE-4.5-0.3B-Base-PT`
        - `kyutai/helium-1-preview-2b`
        - `allenai/OLMo-7B-hf`
        - `mistralai/Ministral-8B-Instruct-2410`
    - MoE（并行支持见下表）
        - `Qwen/Qwen3-30B-A3B`（qwen3_moe，GQA）
        - `mistralai/Mixtral-8x7B-Instruct-v0.1`（mixtral，GQA）
        - `allenai/OLMoE-1B-7B-0924`（olmoe，GQA）
        - `deepseek-ai/DeepSeek-V2-Lite`（deepseek_v2，MLA）
        - `deepseek-ai/DeepSeek-V3`（deepseek_v3，MLA）
        - `zai-org/GLM-4.7`（glm4_moe，MLA）
        - `zai-org/GLM-5`（glm_moe_dsa，MLA + DSA 稀疏注意力）
        - `google/gemma-4-26B-A4B-it`（gemma4_text）

### 注意力

注意力运行在 **FlexInnerAttention** 上。`attn_mask_type` 选择 flex 掩码：
`causal`（纯因果）或 `block_causal`（因果 + 同文档，用于打包 /
SFT 序列）。

### MoE

MoE 块被替换为 TorchTitan 的分组专家 MoE，以启用专家
并行和 `grouped_mm` 快速路径。

并行支持已在调试规模下验证（2 步 loss 下降），覆盖以下
组合。FSDP 可与列出的每个轴组合。

| 模型 | attn | TP | EP | CP | 备注 |
|---|---|---|---|---|---|
| Qwen3-30B-A3B | GQA | 是 | 是 | 是 | 完整矩阵，最高 TP+EP+CP |
| Mixtral-8x7B, OLMoE-1B-7B | GQA | 是 | 是 | 是 | TP+EP，EP+CP |
| DeepSeek-V2-Lite, V3, GLM-4.7 | MLA | 是 | 是 | 是* | TP+EP；*flex+CP 已在 DeepSeek-V2-Lite 上验证（EP+CP） |
| GLM-5 | MLA + DSA | 否 | 是 | 否 | DSA indexer 在 TP 下会显式报错；CP 未接入——仅 FSDP/EP |
| Gemma-4-26B-A4B | GQA | 否 | 是 | 否 | 注意力 TP 不可行（num_global_key_value_heads=2）——仅 FSDP/EP |

PP 尚未接入 MoE 路径（见后续工作）。

## 待解决的已知问题

- 使用 HF 建模时，测试 `FSDP=2 vs FSDP=2 + PP=2` 的 `loss` 和
  `grad_norm` 未按位匹配（但会收敛），而 Torchtitan 建模下是按位匹配的。这将在另一个 PR 中解决，但
  罪魁祸首可能是加载 `seed_checkpoint` 时的 `register_buffer`
- HF 建模的 MFU 低于 Torchtitan MFU

## 后续工作

- MoE 缺少 PP 支持
- 加载 HF 权重
- 添加 LORA 支持
- 支持 Titan RL