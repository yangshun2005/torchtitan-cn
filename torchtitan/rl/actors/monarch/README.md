# Monarch actors

本包在 [Monarch](https://github.com/meta-pytorch/monarch) 上运行独立的 RL 服务。`MonarchActorSystem` 实现与框架无关的 `ActorSystem`（`rl/actors/system.py`），控制器从 `Controller.Config.actor_system`（默认 `MonarchActorSystem.Config()`）构建它，并通过它启动每个服务。自行分配主机的启动器需继承 `Controller` 并重写 `_build_actor_system`，用 `host_meshes` 构建 `MonarchActorSystem`。要从集群调度器获取主机，请使用启动配方（见[启动器](#launchers)）。

每个适配器都是 `Actor`（`actor.py`，一个 Monarch `Actor`）和一个自身不依赖 Monarch 的服务的轻量子类：它把服务的方法声明为 `@concurrent_endpoint`，并提供 Monarch 特有的构造参数（rank、actor 名称、`Channel.open`）。领域行为仍留在服务中。

| 模块 | 用途 |
| --- | --- |
| `system.py` | `MonarchActorSystem`：配置 proc mesh 和 torch-elastic 设置，在各 mesh 上启动每个服务的 actor，按 DP rank 切分 trainer，在 trainer mesh 和每个 actor 的客户端（或 `ModelStateDictTransport`）上初始化 TorchStore，并停止 mesh。 |
| `actor.py` | `Actor`，此处所有 actor 的基类：向已停止或失败的 actor 发送消息时丢弃该消息，而不是报错（Monarch 的默认行为）。 |
| `trainer.py` | `TrainerActor`，`rl/trainer/trainer.py` 中 `Trainer` 的端点。 |
| `generator.py` | `GeneratorActor`，`rl/generator/generator.py` 中 `Generator` 的端点；为数据并行结果汇聚提供 `Channel.open`。 |
| `router.py` | `InterGeneratorRouterActor`，`rl/routing/inter_generator.py` 中 `InterGeneratorRouter` 的端点；强制路由器为单例。 |
| `batcher.py` | `BatcherActor`，`rl/trainer/batcher.py` 中 `Batcher` 的端点，运行在独立的 CPU 进程中；提供 `MonarchRDMAMicrobatch.register`。 |
| `rdma_microbatch.py` | `MonarchRDMAMicrobatch`，基于 Monarch RDMA 缓冲区的 `RDMAMicrobatch`（`rl/trainer/rdma_microbatch.py`）：batcher 持有 microbatch 的大张量，每个 trainer rank 读取自己的列。 |
| `rollout_worker.py` | `RolloutWorkerActor`，控制器主机 CPU 池中的 rollout worker。 |
| `provisioning.py` | Proc mesh：trainer 和 generator 的 GPU mesh（`spawn_proc_mesh`、`PerHostProvisioner`）、控制器主机上的 CPU proc（`spawn_cpu_procs`），以及 Monarch 客户端的 `shutdown`。 |
| `torch_elastic.py` | 在 `init_process_group` 之前，在 proc mesh 上设置 torch-elastic 环境变量（`RANK`、`MASTER_PORT` 等）；rank 0 保持其 rendezvous store 端口打开，以免被其他 socket 占用。 |
| `launcher/` | 调度器启动器：`MonarchLauncher`、`MonarchControllerWithLauncher` 和共享布局规划器（`base.py`）、`SlurmLauncher`（`slurm.py`）、`PBSLauncher`（`pbs.py`），以及提交命令（`submit.py`）。 |

## Launchers

启动器从集群调度器获取作业的主机。启动配方返回 `MonarchControllerWithLauncher.Config`：一个 RL 配方加上在单个集群上运行它的启动器，通过 `MonarchControllerWithLauncher.Config.from_controller_config(config, launcher=...)` 构建。`Controller.Config` 本身没有启动器设置。用 `python -m torchtitan.rl.actors.monarch.launcher.submit` 提交启动配方，它接受与 `torchtitan.rl.train` 相同的参数。作业用这些参数运行 `python -m torchtitan.rl.train`，构建 `MonarchControllerWithLauncher`：它连接到作业的主机，并作为 `host_meshes` 传给 `MonarchActorSystem`。加上 `--dry-run` 可打印规划的节点、角色和设置而不提交。
要支持其他调度器，请在自己的包中继承 `MonarchLauncher`（`launcher/base.py`）。

**布局。** 启动器先放置 trainer，再放置 generators，填满一个节点再填下一个：

- 小于一个节点的角色，若放得下就放在当前正在填充的节点上，否则放到下一个节点。绝不跨两个节点拆分。
- 占用一个或更多节点的角色独占整节点。其 world size 必须是节点 GPU 数的倍数。

以 8-GPU 节点为例（"generators" 为 generator 数量 × 每个 generator 的 GPU 数）：

| Trainer | Generators | 布局 | 空闲 |
| --- | --- | --- | --- |
| 8 GPUs | 8 x 1 GPU | node 0: trainer; node 1: generator_0..7 | 0 |
| 4 GPUs | 3 x 4 GPUs | node 0: trainer, generator_0; node 1: generator_1, generator_2 | 0 |
| 16 GPUs | 2 x 8 GPUs | nodes 0-1: trainer; node 2: generator_0; node 3: generator_1 | 0 |
| 6 GPUs | 2 x 4 GPUs | node 0: trainer; node 1: generator_0, generator_1 | node 0 上空闲 2 GPUs |
| 4 GPUs | 1 x 8 GPUs | node 0: trainer; node 1: generator | node 0 上空闲 4 GPUs |
| 12 GPUs | 1 x 4 GPUs | `ValueError`：12 不是 8 的倍数 | |

### SLURM

提交前，按[快速开始](../../README.md#quick-start)设置环境、模型和数据集，并确保每个节点都能在相同路径访问它们（详见 [`SlurmLauncher` docstring](./launcher/slurm.py)）。在仓库根目录、venv 已激活且已设置 `PYTHONPATH` 的 shell 中提交。JIT 缓存（Triton、vLLM、FlashInfer）默认位于节点本地 `/tmp/torchtitan-<uid>`；可在 `launcher.env` 或提交 shell 中设置以覆盖。

从登录节点执行以下命令，可在两个 4-GPU 节点上运行完整示例：trainer 在 node 0，四个 1-GPU generator 共享 node 1（将配方中的 `gpus_per_node` 设为节点的 GPU 数）：

```bash
python -m torchtitan.rl.actors.monarch.launcher.submit \
  --module torchtitan_recipes.rl.alphabet_sort --config rl_grpo_qwen3_0_6b_flex_slurm
```

账户、分区等站点设置放在自己的启动配方的 `slurm_args` 中：

```python
def my_rl_slurm():
    config = rl_grpo_qwen3_0_6b_flex_slurm()
    config.launcher.slurm_args += ["--account=my-account", "--partition=gpu"]
    return config
```

### PBS

`PBSLauncher` 的提交方式相同。它通过 `qsub -V` 传递提交环境；队列、项目、walltime 等站点设置放在 `pbs_args` 中，Cray PALS 集群设置 `mpiexec` 选项。启动配方用它包装 RL 配方：

```python
def my_rl_pbs():
    return MonarchControllerWithLauncher.Config.from_controller_config(
        rl_grpo_qwen3_0_6b_flex(),
        launcher=PBSLauncher.Config(
            gpus_per_node=4,
            pbs_args=["-q debug", "-A my_project", "-l walltime=01:00:00"],
            mpiexec_args=["--cpu-bind=none"],
        ),
    )
```

每个节点都会启动 `python_exe`，因此在用 module 设置环境的集群上，请将其指向一个先加载 module 再运行 `exec python "$@"` 的脚本。该脚本也是将 `TMPDIR` 设为 `/tmp` 这类短路径的地方：PBS 将其设为 `/var/tmp/pbs.<job id>/...`，会使 Monarch 的 unix socket 路径超过 Linux 允许的长度（meta-pytorch/monarch#5415；修复后可去掉此项）。

启动器的 `env` 值属于配置，控制器可能会记录并保存（`print_config`、`save_config_file`），因此不要在其中放密钥；请在提交 shell 中设置，作业会继承。

## 本包之外

本包之外没有模块导入 Monarch 或本包，除了控制器 `Controller.Config.actor_system` 的默认值和启动配方（它们延迟导入）。控制器、rollouter 和 `train.py` 使用抽象的 `ActorSystem`。

剩下的外部约定是 actor system 返回的句柄上的调用方式（见 `ActorSystem`）：`.call()` 及其结果上的 `.get(rank)`、`.call_one()`、`.choose()`、`.stream()` 和 `.broadcast()`。trainer 以 `TrainerHandle` 返回，包含全 rank、rank-0 和按 DP rank 的句柄；每个 generator 以 `ServiceHandle` 返回给路由器，因此调用方无需自己切分 mesh。generator 句柄只来自路由器：其 `reserve` 返回 generate 调用所指向的句柄，其他所有 generator 调用都经过路由器的端点。这些调用出现在控制器（及其 `GenerateFn`）、rollouter（对 worker 池的 `.choose()`）和路由器（对 generator 句柄）中。batcher 和 trainer 仅通过 `RDMAMicrobatch` 接口看到 RDMA microbatch。

其他 actor 框架（如 Ray）会添加自己的 `ActorSystem` 实现和适配器，包装其句柄以提供此调用约定，并通过 `Controller.Config.actor_system` 选择。