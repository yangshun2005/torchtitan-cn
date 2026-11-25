# 使用 Verifiers 的 DAPO Math

本示例保持现有 [DAPO Math](../../dapo_math) 配方不变，仅将其 rollout 路径替换为 [Verifiers](https://github.com/PrimeIntellect-ai/verifiers)。

它使用 Verifiers 的 null harness 和本地子进程运行时，执行单轮数学任务。不提供工具或沙箱，因此不要用此配置执行不受信任的代码。

## 环境准备

按照 [TitanRL 环境准备](../../../README.md)操作，然后安装本示例的依赖：

```bash
pip install -r torchtitan/rl/examples/verifiers/dapo_math/requirements.txt

python scripts/download_hf_assets.py \
  --repo_id Qwen/Qwen3-4B-Base \
  --all
```

## 运行

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.verifiers_dapo_math \
  --config rl_dapo_qwen3_4b_verifiers_8k
```

32K 响应变体请使用 `rl_dapo_qwen3_4b_verifiers_32k`。