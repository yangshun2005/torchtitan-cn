## 配置

一次运行由**完整配置**描述：一个返回完整 `Trainer.Config` 的函数——包含模型、并行度和所有优化选项。配置用 Python 编写，因此新建配置就是用 TorchTitan 组件进行配置编程。

用 `--module`（定义该函数的模块）和 `--config`（该函数）选择配置：

```bash
NGPU=4 MODULE=tests.integration_tests.features CONFIG=llama3_debugmodel_fsdp2_cp2 ./run_train.sh
```

并行度写在配置里，但 world size 不在。因此 `NGPU` 仍须等于各并行度的乘积。要改变任何行为，直接改配置——自己写函数返回新的 `Trainer.Config`，而不是用 CLI 标志：

```python
# torchtitan_recipes/my_runs.py
def llama3_debugmodel_fsdp2_cp4() -> Trainer.Config:
    config = llama3_debugmodel_fsdp2_cp2()
    config.parallelism.context_parallel_degree = 4
    config.training.steps = 100
    return config
```

命令行刻意不暴露通用的 `--section.option` 配置。把训练行为写进 recipe，这样一次运行就能凭其模块名和函数名复现。

支持的操作选项：

- `--override TARGET[=JSON]`，可重复，用于已注册组件的覆盖
- `--comm-backend BACKEND`，用于选择启动环境的通信后端
- `--output-dir PATH`，用于指定本次运行的输出位置
- `--resume-step STEP`，用于从检查点恢复运行
- `--print-config`，打印解析后的配置并退出

### 配置存放位置

[torchtitan_recipes](../../torchtitan_recipes/) 包存放完整配置——一个 recipe 就是其中一个函数。它与 `torchtitan` 并列而非置于其内，因为两者存放的东西不同：`torchtitan` 提供模型定义和实现各项优化的类；`torchtitan_recipes` 只挑选这些组件的组合。配置还与某个集群、某次运行绑定，因此其变更节奏与库不同，发布一个配置与发布一个类也不是同一种承诺。

`torchtitan_recipes` 是第二个顶层包，因此在其存在之前做的可编辑安装不认识它，在 torchtitan 仓库根目录之外执行 `import torchtitan_recipes` 会失败。若在仓库根目录外运行，重新执行一次 `pip install -e .`。在仓库根目录运行（如 `run_train.sh` 和 CI 那样）则两种情况下都能工作。

### 编写自己的配置

不同集群通常意味着不同的分片布局，因而需要不同的配置。这无需改代码：在 `torchtitan_recipes/models/` 下、以模型命名的模块中加一个函数，并在命令行指定它。测试配置单独存放：可复用的模型测试夹具放在 `torchtitan_recipes/tests/models/` 下，跨模型集成配置则放在 `tests/integration_tests/` 中其测试定义旁边。

```python
# torchtitan_recipes/models/llama3.py
def llama3_8b_fsdp8_tp2_h200() -> Trainer.Config:
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=8,
        tensor_parallel_degree=2,
    )
    model_config = build_model_config(
        "8B", attn_backend="flex", parallelism=parallelism
    )
    return Trainer.Config(
        model=model_config,
        parallelism=parallelism,
        ...
    )
```

模型配置是针对某一种并行度构建并分片的，因此先声明并行度，再传给 `build_model_config`。从另一个 recipe 派生、且改变了序列并行或专家并行的 recipe，必须为新并行度构建模型（例如通过基础 recipe 的 `parallelism=` 参数）。

`--module` 接受任何可导入的模块，因此存放在本仓库之外的配置同样可用：

```bash
MODULE=my_company_configs.experiments CONFIG=llama3_ablation_7 ./run_train.sh
```

### 配置优化器

优化器配置是有类型的，且刻意只暴露经过验证的 TorchTitan recipe 所用的选项。例如 `Adam.Config` 和 `AdamW.Config` 都暴露 `weight_decay`。它们不会把不受限的关键字参数字典转发给 PyTorch 优化器。需要更多上游选项时，给对应的优化器配置加一个类型化字段；这样 recipe 保持可验证，其支持的行为也明确可见。

### 转换模型配置

[模型配置转换](transform/README.md)是修改已完成模型配置的一种受支持方式。它们是可选的，用户也可以用自己的工具。

### 校验

只涉及单个配置字段的简单不变量，放在该配置的 `__post_init__` 中。跨配置分节的检查放在 [validation.py](validation.py) 中。`Trainer.Config.__post_init__` 在完成本地校验后调用这些检查。

### 添加配置字段

新字段应归入配置树：放在它所配置的组件上、模型上，或者——对于少数没有其他归属的选项，如 `training.num_tokens_per_microbatch_per_dp_rank`——在与维护者讨论后放入 [configs.py](configs.py)。添加 dataclass 字段不会改变 CLI 接口。

需要自有选项的实验，应继承 `Trainer` 及其嵌套的 `Config`，并在其 recipe 中返回该类型：

```python
# my_project/trainer.py
@dataclass(kw_only=True, slots=True)
class CustomConfig:
    how_is_your_day: str = "good"


class MyTrainer(Trainer):
    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        custom_config: CustomConfig = field(default_factory=CustomConfig)


# my_project/recipes.py
def my_experiment_debugmodel() -> MyTrainer.Config:
    return MyTrainer.Config(custom_config=CustomConfig(how_is_your_day="great"))
```

```bash
MODULE=my_project.recipes CONFIG=my_experiment_debugmodel ./run_train.sh
```

### `torchtitan_recipes` 中应放什么

本仓库刻意只发布一小部分：

- `tests/`——CI 使用的模型夹具和组合配置
- 经维护的训练、收敛、性能或功能工作流验证过的配置

我们不发布模型、并行度和优化的所有组合，因为该集合是指数级的。你的运行就是你自己的配置：可以加在这里但不提交，或放在自己的包中并用 `--module` 指向它。从已发布的配置派生只需几行，如上所示。