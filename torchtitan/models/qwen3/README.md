## 可用功能
#### 稠密模型
- Qwen3 稠密模型：
    - 支持 FSDP/HSDP、TP、CP、DDP。
    - 支持 AC、torch.compile。
- Qwen3 MoE 模型：
    - 支持 FSDP/HSDP、TP、CP、DDP、EP。
    - 支持 AC、torch.compile。
    - MoE 模型使用 Token Choice 路由。负载均衡遵循原始 Qwen3 训练配方（辅助损失）。

稠密和 MoE 调试模型由 `tests/integration_tests/models.py` 覆盖
（FSDP+TP+CP，MoE 为 FSDP+TP+CP+EP）。

模型架构包含 4B、8B 和 235B-A22B，但这些规模尚未有经过验证的训练配方。

## 下载 Qwen3 分词器
```bash
python scripts/download_hf_assets.py --repo_id <hf_repo_name> --assets tokenizer
```

例如，0.6B 和 1.7B 仓库分别为 `Qwen/Qwen3-0.6B` 和
`Qwen/Qwen3-1.7B`。每个受支持规格的常规本地目录为
`assets/hf/<repository basename>`。


## 剩余工作
- 为 4B、8B 预训练和 235B-A22B 添加经过验证的配方。
- 在更长的训练任务上验证学习率和调度，或引用官方
  参考资料。
- 与已有的性能基准进行对比。