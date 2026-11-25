# 低精度训练

TorchTitan 提供以下低精度训练格式：

- [MXFP8](mxfp8/README.md)：用于 Blackwell GPU 上的稠密模型和 MoE 模型。
- [NVFP4](nvfp4/README.md)：用于 Blackwell GPU 上的实验性稠密训练。

## TorchAO 与 TorchTitan 的边界

TorchTitan 负责这些格式的训练集成：模块、自动微分、配方选择、分布式组合，以及量化权重操作数的生命周期。持久化模型参数和检查点状态仍保持高精度。

TorchAO 提供底层量化、布局变换和矩阵乘法内核。这一边界让 TorchTitan 能将低精度训练与 FSDP 及其他并行方式集成，同时 TorchAO 可独立演进并优化底层内核。