# 核心发布流程

本文档说明如何暂存、验证并发布稳定的 TorchTitan 版本。

示例使用 TorchTitan `0.3.0`、PyTorch `2.14.0`、torchvision `0.29.0` 和 torchao `0.18.0`。每个发布列车都需更新所有版本号和 wheel 索引 URL。

## 何时发布

- 跟随每个 PyTorch 次版本发布，约每两个月一次：在 PyTorch 创建 `release/X.Y` 后不久切出对应的 TorchTitan 发布分支，然后随 PyTorch 稳定版一起发布 TorchTitan。
- 紧急补丁时，从现有 `release/0.Y` 分支发布 `0.Y.(Z+1)`，无需新的 PyTorch 次版本。

## 步骤 1 - 验证 `main` 并创建发布分支

切分支前，完成 CI 验证和发布专项测试。

### 1A - 确认所有必需的 CI 均通过

检查每个适用工作流最新的 `main` 或定时运行，包括：

- lint 和 CPU/GPU 单元测试；
- 8-GPU Real-PG 功能和模型集成测试；
- H100 和 B200 集成测试；以及
- `torchtitan/experiments/` 下项目的集成测试，包括 GraphTrainer、Transformers 建模后端、RL 和 TorchFT。

按路径过滤的实验工作流可能不会在每次 `main` 提交上运行，因此需显式检查其最新定时运行。切发布分支前调查所有失败。CI 必须通过，但仅凭 CI 状态不足以判定可发布。

### 1B - 创建干净的开发环境

```bash
cd ~/torchtitan
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e . -r requirements.txt -r requirements-dev.txt
```

安装 PyTorch 发布暂存包。更新所测版本的版本号和 CUDA wheel 索引。

```bash
uv pip install --force-reinstall --pre \
  --index-url https://download.pytorch.org/whl/test/cu130 \
  "torch==2.14.0" \
  "torchvision==0.29.0" \
  "torchao==0.18.0"
```


### 1C - 运行单元测试和冒烟测试

```bash
pre-commit run --all-files
pytest tests/ -x
```

对仓库中记录的示例和教程命令做冒烟测试。确认说明准确，且每个选定的任务都能完成。

### 1D - 选择版本并创建分支

第一个候选版本用 `0.Y.0rc1`，最终版本用 `0.Y.0`。在
[PyPI 发布历史](https://pypi.org/project/torchtitan/#history)中查看最新已发布版本。

```bash
git checkout main
git pull origin main
git checkout -b release/0.3
git push -u origin release/0.3
```

## 步骤 2 - 设置 RC 版本

将 [`assets/version.txt`](assets/version.txt) 编辑为 RC 版本，然后针对发布分支（而非 `main`）开 PR。

基础 `0.Y.Z` 版本遵循[语义化版本](https://semver.org/)：

- 功能发布递增 `Y`，例如 `0.3.0` -> `0.4.0`；
- 包含修复的补丁发布递增 `Z`，例如 `0.3.0` -> `0.3.1`；以及
- 候选版本追加 Python RC 后缀 `rcN`，例如 `0.3.0rc1`，如需另一个候选版本则为 `0.3.0rc2`。

```bash
git checkout release/0.3
git pull origin release/0.3
echo "0.3.0rc1" > assets/version.txt
```

开一个目标为 `release/0.3` 的 PR，等待 CI，然后合并。

## 步骤 3 - 在 TestPyPI 暂存 RC

### 3A - 更新固定的发布依赖

暂存新版本前：

1. 在
   [`.github/workflows/validate_rc.yaml`](.github/workflows/validate_rc.yaml)
   中更新固定的 `torch`、`torchvision`、`torchao` 和 `triton` 版本。将 Triton 设为所选 PyTorch wheel 要求的版本。
2. 在 `.github/workflows/validate_release_gpu.yml` 中更新固定的 `torch`、
   `torchvision` 和 `torchao` 版本以及 PyTorch wheel 索引。Triton 由 GPU PyTorch wheel 传递安装，在运行时验证，不单独固定。
3. 暂存 RC 前，将两个工作流更新合并到发布分支。

### 3B - 将 RC 发布到 TestPyPI

[`test_release.yml`](.github/workflows/test_release.yml) 工作流构建 wheel 和源码分发包，运行 `twine check --strict`，并将构建产物上传到 TestPyPI。构建前，它会对所选发布分支上的所有文件调用可复用的 lint 工作流。

运行方式：

1. 打开 **GitHub Actions -> Publish a Release to TestPyPI**。
2. 点击 **Run workflow**。
3. 选择发布分支，例如 `release/0.3`。
4. 启动工作流。
5. 确认 lint 和构建任务通过。
6. 出现提示时，打开 **Review deployments**，选择受保护的 `test-release` 环境，点击 **Approve and deploy**。

### 3C - 确认暂存的发布版本

确认 RC 出现在
[TestPyPI 发布历史](https://test.tori.org/project/torchtitan/#history)中。在包可见且可安装之前，不要继续。

## 步骤 4 - 验证 RC 安装

### 4A - 自动 CPU 包验证

TestPyPI 暂存工作流在发布后自动调用 `validate-rc` 任务。它不会在 push 或定时触发时运行。

对于 Python 3.11 和 3.12，该任务创建干净的虚拟环境并：

- 从 PyTorch 测试 CPU 频道安装固定的 `torch`、`torchvision`、`torchao` 和 `triton` 包；
- 从 TestPyPI 安装精确的 TorchTitan RC；
- 验证 TorchTitan 版本并确认其从 `site-packages` 导入；
- 运行简短的 CPU 前向、反向和优化器冒烟测试；
- 验证 loss 有限且下降；以及
- 在隔离的临时目录中，针对已安装的 RC 运行所有启用的 CPU 单元测试。

确认每个 `validate-rc` 矩阵任务均通过。

可选的手动 GPU 安装检查：

```bash
python -m pip install --pre \
  --index-url https://download.pytorch.org/whl/test/cu130 \
  "torch==2.14.0" \
  "torchvision==0.29.0" \
  "torchao==0.18.0"

python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple/ \
  "torchtitan==0.3.0rc1"

python - <<'PY'
import torch
import torchtitan

assert torch.cuda.is_available()
print(f"torchtitan={torchtitan.__version__} ({torchtitan.__file__})")
print(f"torch={torch.__version__}, cuda={torch.version.cuda}")
print(f"gpu={torch.cuda.get_device_name(0)}")
PY
```

### 4B - 按需 GPU RC 验证

RC 在 TestPyPI 可用后，从发布分支运行 **Validate a TestPyPI Release Candidate on GPUs** 工作流。每个任务都从 TestPyPI 安装精确的 TorchTitan RC，并搭配固定的 PyTorch 发布暂存包。

运行方式：

1. 打开 **GitHub Actions -> Validate a TestPyPI Release Candidate on GPUs**。
2. 点击 **Run workflow**。
3. 选择发布分支，例如 `release/0.3`。
4. 启动工作流，等待所有验证任务完成。

该工作流运行七个 GPU 任务：四个在 8x A10G 运行器上，两个在 8x H100 运行器上，一个在 8x B200 运行器上。所有集成套件均使用真实进程组：

- `validate-standard-gpu` 搭配 `core` 套件，检查已安装的 RC 和依赖版本、所有启用的单 GPU 和多 GPU 单元测试、Real-PG 功能和模型套件及其集成的 loss 和梯度范数基准，以及 Flux；
- `validate-standard-gpu` 搭配 `graph-trainer` 套件，运行标准 GraphTrainer 集成、数值、图 pass、profiler、追踪、预编译、位级确定性和 SAC 峰值内存测试；
- `validate-standard-gpu` 搭配 `torchft` 套件，安装固定的稳定版 `torchft==0.2.0`，启动 Lighthouse，并在启用检查点的情况下运行 8-GPU TorchFT 集成测试 10 个训练步；
- `validate-standard-gpu` 搭配 `transformers-modeling-backend` 套件，安装 `transformers==5.9.0`，运行 MoE FSDP+TP+EP+CP、稠密 FSDP+TP+PP、稠密 CP+PP 和 SFT 集成测试；
- `validate-h100` 运行基础 H100 套件、DeepEP 激活检查点梯度检查、Qwen3 搭配 DeepEP v2，以及 DeepSeek V3 搭配 HybridEP。每个 DeepEP 变体在安装其匹配依赖后测试；
- `validate-graph-trainer-h100` 运行 H100 GraphTrainer 集成、MoE 数值、DeepSeek V3 预编译和位级确定性测试；
- `validate-b200` 运行 Dist-MoE CPU/GPU 和 MXFP8 FSDP 单元测试。它运行完整的 B200 集成套件：Kimi K3 多模态、DistMuon 数值、流水线并行、MXFP8/NVFP4 线性层，以及搭配 VMM 和流水线并行的 BF16/MXFP8 Dist-MoE。公共 CUDA 容器使用干净的 uv 环境，在验证固定包版本前安装 FlashAttention 4 和 Dist-MoE。

每个 GPU 测试任务都从 `site-packages` 安装并导入精确的 TestPyPI RC，并检查预期的 `torch`、`torchvision` 和 `torchao` 版本。RL 和 GraphTrainer AutoParallel 有意不包含在内。ROCm 和单独的 Fake-PG 执行模式不在本 GPU 工作流覆盖范围内。

### 4C - 发布验收标准

定版前，确认：

- 步骤 4A 的 **Publish a Release to TestPyPI** 工作流通过，包括其 Python 3.11 和 3.12 的 `validate-rc` 任务；
- 步骤 4B 的 **Validate a TestPyPI Release Candidate on GPUs** 工作流通过，包括 `Validate core on 8x A10G`、`Validate graph-trainer on 8x A10G`、`Validate torchft on 8x A10G`、`Validate transformers-modeling-backend on 8x A10G`、`Validate core H100 tests`、`Validate GraphTrainer H100 tests` 和 `Validate core B200 tests`；
- 在 GPU 任务日志中搜索 `SKIPPED` 和 `Skipping test`，仅发现预期的排除项。

如果暂存的包需要代码修复，按步骤 5 操作，递增 RC 版本，发布新 RC，并重复验证。切勿复用 RC 版本。

## 步骤 5 - 排查 RC 验证失败

更改发布前先确定根因。修复路径取决于问题出在 TorchTitan、PyTorch、torchao 还是验证环境。

### 5A - Bug 在 TorchTitan 中

先在 `main` 上修复，然后将合并后的修复 cherry-pick 到发布分支。

1. 将修复 PR 合入 `main`，记录其合并提交 SHA。
2. 将该提交 cherry-pick 到发布分支：

   ```bash
   git checkout release/0.3
   git pull origin release/0.3
   git cherry-pick <sha>
   # 如有必要解决冲突，然后推送分支。
   git push origin release/0.3
   ```

3. 将 `assets/version.txt` 更新为下一个 RC，例如 `0.3.0rc2`。
4. 重复步骤 3-4 发布并验证新 RC。切勿复用 RC 版本。

如果紧急修复无法等待 `main` CI，直接针对 `release/0.3` 开修复 PR，然后将相同更改前向移植到 `main`。

### 5B - Bug 在 PyTorch 中

#### PyTorch 尚未发布

1. 在 `pytorch/pytorch` `main` 中合入修复，或与 PyTorch 开发者合作合入。
2. 将其标记为 `X.Y` 的发布阻塞项，并请求 cherry-pick 到 `pytorch/pytorch` `release/X.Y`。发布经理必须在 cherry-pick 截止日期前批准。
3. PyTorch 发布下一个 RC 时，更新固定版本并重复步骤 3-4。
4. 如果上游修复无法及时合入，使用有文档记录的 TorchTitan 临时方案，并在上游修复可用后移除。

#### PyTorch 已正式发布

已发布的 wheel 无法更改。提交上游 issue，等待 PyTorch 补丁发布（如 `X.Y.1`），然后更新固定版本并重新验证。如果发布无法等待，使用有文档记录的 TorchTitan 临时方案。

同样的策略适用于 torchao 问题。

## 步骤 6 - 定版

所有 CPU 和 GPU RC 验证通过后：

1. 在发布分支上更新 [`assets/version.txt`](assets/version.txt)：

   ```bash
   git checkout release/0.3
   git pull origin release/0.3
   echo "0.3.0" > assets/version.txt
   ```

2. 开一个目标为 `release/0.3` 的 PR。
3. 确认 CI 通过并合并 PR。

## 步骤 7 - 创建 GitHub Release 并发布到 PyPI

1. 打开[新发布页面](https://github.com/pytorch/torchtitan/releases/new)。
2. 将标签设为 `v0.3.0`，目标分支设为 `release/0.3`。
3. 点击 **Generate release notes**。验证 **Full Changelog** 对比的是上一个发布标签，然后整理更改，并添加固定的 torch 和 torchao 版本以及简短亮点摘要。
4. 最终稳定版不要勾选 **Set as a pre-release**。
5. 点击 **Publish**。这会触发
   [`.github/workflows/release.yml`](.github/workflows/release.yml)。
6. 出现提示时，批准受保护的 `release` 环境部署。

## 步骤 8 - 验证生产 PyPI 发布

使用干净环境并验证实际安装的 wheel：

```bash
python -m venv /tmp/torchtitan-release-verify
source /tmp/torchtitan-release-verify/bin/activate
python -m pip install --upgrade pip

python -m pip install \
  --index-url https://download.pytorch.org/whl/cu130 \
  "torch==2.14.0" \
  "torchvision==0.29.0" \
  "torchao==0.18.0"

python -m pip install "torchtitan==0.3.0"

# 在仓库外运行，避免检出目录遮蔽已安装的 wheel。
cd /tmp
python - <<'PY'
import importlib.metadata
import sysconfig
from pathlib import Path

import torchtitan

expected_version = "0.3.0"
package_path = Path(torchtitan.__file__).resolve()
site_packages = Path(sysconfig.get_paths()["purelib"]).resolve()

assert importlib.metadata.version("torchtitan") == expected_version
assert torchtitan.__version__ == expected_version
package_path.relative_to(site_packages)

print(f"torchtitan={torchtitan.__version__}")
print(f"package_path={package_path}")
PY
```

然后：

1. 确认发布出现在
   [PyPI 发布历史](https://pypi.org/project/torchtitan/#history)中。
2. 确认验证命令成功退出，并打印
   `torchtitan=0.3.0`，且 `package_path` 位于
   `/tmp/torchtitan-release-verify/lib/python*/site-packages/` 下。
3. 针对已安装的 PyPI wheel 运行简短的 debug 模型训练检查。验证 loss 有限且下降。仅做导入检查不够。

## 发布验证覆盖范围

### 当前验证层

1. **`main` 上的切分支前 CI：** lint、CPU/GPU 单元测试、Real-PG 集成测试、H100/B200 测试，以及 `torchtitan/experiments/` 下项目的最新可用 CI。
2. **发布专项源码验证：** 本地单元和冒烟测试，以及针对固定 PyTorch 发布暂存包的 Real-PG 功能/模型套件及其集成数值检查。
3. **TestPyPI 暂存和 CPU 验证：** 发布前运行全仓库 lint；干净的 Python 3.11 和 3.12 环境随后安装暂存的 RC，验证包版本和导入位置，运行简短训练步，并运行完整的已启用 CPU 单元测试套件。
4. **TestPyPI GPU 验证：** 七个 GPU 任务，四个在 8x A10G 运行器上，两个在 8