# DeepSeek V4

本文件夹包含 DeepSeek V4 的 TorchTitan 实现。

模型入口为 `torchtitan.models.deepseek_v4.build_model_config`。CI 使用的调试训练配置位于 `torchtitan_recipes.tests.models.deepseek_v4`：

- `deepseek_v4_debugmodel`
- `deepseek_v4_mtp_debugmodel`

`deepseek_v4_flash` 和 `deepseek_v4_pro` 架构变体仍可通过 `build_model_config` 使用；未提供经过验证的完整训练配方。

## 组件

- `model.py`：解码器模型和 transformer 块定义。
- `attention.py`：DeepSeek V4 稀疏注意力变体，包括滑动窗口注意力、高度压缩注意力和压缩稀疏注意力。
- `compressor.py`：KV 压缩和稀疏索引选择辅助函数。
- `mhc.py`：多 token/头耦合分支混合模块。
- `moe.py`：DeepSeek V4 MoE 路由器和专家封装。
- `sharding.py`：TP、SP、EP、FSDP、DTensor 和 `spmd_types` 的声明式分片配置。
- `state_dict_adapter.py`：DeepSeek V4 的检查点键映射。

## 冒烟测试

在 4 块 GPU 上使用 FSDP4 和 EP4 运行 DistMuon + AdamW 调试模型：

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 \
  MODULE=torchtitan_recipes.tests.models.deepseek_v4 CONFIG=deepseek_v4_debugmodel_fsdp4_ep4 \
  ./run_train.sh
```

## 状态

AdamW 调试模型已在 4 块 GPU 上使用 FSDP2、TP2、EP2 和 `spmd_types` 后端完成冒烟测试。CI 在 8 块 GPU 上使用 FSDP2、TP2、EP2 和 Interleaved1F1B PP2 进行训练。流水线并行需要真实进程组，不支持 MTP 层。可选的 MTP 路径通过 `deepseek_v4_mtp_debugmodel` 使用。在添加生产配方之前，应完成检查点兼容性和更大规模收敛验证。