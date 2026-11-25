欢迎社区提交可复现的基准测试结果。

## 提交指南

提交内容应为一个或多个文件，包含以下信息：

1. 实体：可以是你的姓名、GitHub 用户名、公司、大学、团队等。
2. 基准测试的模型或主题，例如 Llama 3.1、Async TP。
3. 硬件配置，包括 GPU 类型、互联方式等。
4. 实际性能报告及训练配置，例如通过以下方式提供：
   - Python 配置文件 / 命令行参数
   - 完整的解析后配置，通过向 `run_train.sh` 传入 `--print-config` 打印获得（推荐，因为默认值可能随时间变化）
5. `torchtitan`、`torch`、`torchao` 或任何相关依赖的版本及日期/时间。
6. 其他有助于复现结果的说明。

文件名应遵循以下格式：
```
[model/theme]_[hardware]_[date/time]_[entity].md
```
例如 `llama3.1_h100_202412_pytorch.md`、`asynctp_256xh100_20250613_alice+bob.md`。

示例见 [llama3_h100_202412_torchtitan.md](./llama3_h100_202412_torchtitan.md)。