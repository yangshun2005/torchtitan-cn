# 测试检查点转换的正确性

在文件类型或模型定义之间转换检查点时，需要确保转换后的检查点正确，即模型定义保持不变，包括转换后检查点的权重在新目标程序上下文中加载时能产生相同输出。

本指南提供测试转换脚本正确性的通用框架。这里使用的示例是 HuggingFace 与 `torchtitan` 之间的双向转换。

## 方法

### 健全性检查（贪心解码）
快速检查转换是否正确的方法：对初始检查点和转换后检查点分别执行贪心解码推理，确认结果相同。此方法不保证正确性，但若模型定义不一致，很可能快速得到**真阴性**结果。[`generate.py`](/torchtitan/rl/generate.py)（vLLM 或 SGLang）基本与模型无关，可用于此目的；传入 `--temperature 0` 启用贪心解码。该方法已在部分模型上测试，并非全部。对于没有自身推理入口的模型，方法同样适用。

注意：模型定义需与转换脚本匹配。例如，从 `torchtitan` 转换到 HuggingFace 时，务必包含与 `torchtitan` 模型架构匹配的正确 `config.json` 文件。用 HuggingFace `transformers` 加载模型时提供错误的 `config.json`，即使权重转换正确，也会导致生成结果错误。

### 全面检查（KL 散度）
在 `./scripts/checkpoint_conversion/numerical_tests_example.py` 中，将对 `torchtitan` 中加载的 DCP 检查点和 HuggingFace `AutoModelForCausalLM` 中加载的 safetensors 检查点执行前向计算。该脚本测试 HuggingFace -> `torchtitan` 方向，因为加载 HuggingFace 检查点需要：
- 将实例化的 `torchtitan` state dict 通过 `to_hf` 转换，以便加载 safetensors 权重；
- 通过 `from_hf` 将 HF 版本的 state dict 转换回 torchtitan。

在 HuggingFace 与 `torchtitan` 之间转换 Llama 3 时，需对若干注意力矩阵执行置换，以弥补 HuggingFace 与原生 Llama RoPE 实现的差异。为演示 KL 散度测试如何揭示此类细微误差，我们额外比较了原始模型与转换后模型在有/无置换情况下的 KL 散度。结果如下：
```
$ python ./scripts/checkpoint_conversion/numerical_tests_example.py
Average loss of test from_hf is -1.45365707318601e-13
Average loss of test from_hf_no_perm is 5.368335223465692e-06
```