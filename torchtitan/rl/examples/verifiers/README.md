# Verifiers 集成

本目录包含 TitanRL 与 [Verifiers](https://github.com/PrimeIntellect-ai/verifiers) 之间的可选桥接层。其他 TitanRL 配方不导入也不依赖 Verifiers。

职责边界如下：

- TitanRL 负责加载样本、调度 rollout 组、路由生成、跟踪策略版本、计算优势值并训练策略。
- Verifiers 负责运行环境、agent 或工具循环、运行时以及任务奖励。
- 适配器负责在两个系统之间转换样本和轨迹。

可复用的桥接代码与本 README 放在一起。每个实验有独立的包和依赖：[`dapo_math/`](./dapo_math)（单轮数学）和 [`terminal_bench/`](./terminal_bench)（基于 Harbor 任务的多轮终端 agent）。

## 集成流程

一个组，自上而下。每一列是一个进程，或 agent 在其运行时中的执行。

```text
     controller      rollout worker            env server     agent        router     generator
     (Rollouter)     (VerifiersRolloutWorker   (Verifiers)    (runtime)
                     and its GenerationServer)
          |                  |                      |              |            |           |
 1        |--run_group------>|                      |              |            |           |
          |                  |                      |              |            |           |
 2        |                  |--task, per rollout-->|              |            |           |
          |                  |                      |              |            |           |
 3        |                  |                      |--start------>|            |           |
          |                  |                      |              |            |           |
 4        |                  |                      |<--chat-------|            |           |
          |                  |                      |              |            |           |
 5        |                  |<--token ids----------|              |            |           |
          |                  |                      |              |            |           |
 6        |                  |--reserve---------------------------------------->|           |
          |                  |--generate---------------------------------------------------->|
          |                  |                      |              |            |           |
 7        |                  |--tokens, logprobs--->|--reply------>|            |           |
          |                  |                      |              |            |           |
 8        |                  |                      |--grade------>|            |           |
          |                  |                      |              |            |           |
 9        |                  |<--trace, reward------|              |            |           |
          |                  |--release_session------------------------------->|           |
          |                  |                      |              |            |           |
10        |<--scored group---|                      |              |            |           |
```

1. 控制器向 rollout worker 发送一个任务。任务是来自 [`data.py`](./data.py) 的纯数据。
2. worker 在组的每次 rollout 中向 env server 发送一次任务：一个普通 dict，通过 ZMQ 以 msgpack 传输（[`env_server.py`](./env_server.py)）。
3. env server 在环境的运行时中启动 agent，例如本地进程或 Docker 容器。若使用池，则由池中的一个 worker 执行该 rollout。
4. 每一轮 agent 都向 env server 中 Verifiers 的拦截服务器发送聊天请求。Verifiers 的训练客户端将消息渲染为 token id。
5. token id 通过 HTTP 到达 rollout worker 的 [`GenerationServer`](./generation_server.py)，地址在控制器主机上。
6. GenerationServer 调用 TitanRL 的 `generate_fn`，后者在 router 上预留一个 generator，然后调用该 generator。
7. GenerationServer 返回 token 和 logprobs，并保留本次调用的策略版本和 generator 指标：轨迹中没有对应字段。agent 收到回复。第 4 到 7 步每轮重复。
8. agent 停止后，Verifiers 对 rollout 评分，例如运行任务的测试。
9. env server 返回该 rollout 的轨迹。worker 在 router 上释放每条轨迹的路由会话。
10. worker 将 rollout 的可训练轨迹转换为 TitanRL rollout 轮次（[`rollouter.py`](./rollouter.py)），对组评分、计算优势值并返回。

进程数量：控制器、rollout worker（每个 `rollouts_per_worker` 个在途训练 rollout 一个，向上取整），以及所有 worker 共享的一个 env server。若使用池（`verifiers_env_server.serve.pool`），env server 在其池 worker 中执行 rollout。因此 4 个 rollout worker 加 8 个池 worker 会给控制器增加 4 + 1 + 8 个进程，而非 4 x 8。env server 和每个池 worker 首先运行 [`worker_setup.py`](./worker_setup.py)，因此该模块不得导入 torch：导入 torch 会让每个进程多占约 0.5 GB 内存（`import torch` 后的峰值 RSS）。

taskset 只在训练数据集中设置一次（`VerifiersTaskDataset.Config.verifiers_taskset`）。`VerifiersRollouter.Config` 会据此填充 env server 的 taskset，因此不要设置 `verifiers_env_server.environment.taskset`。

## 添加实验

对于任务 `my_task`：

```text
torchtitan/rl/examples/verifiers/my_task/
  __init__.py
  data.py                                  # 你的 Taskset 及其 TasksetConfig
  requirements.txt                         # 其依赖
torchtitan_recipes/rl/verifiers_my_task.py # 返回 Controller.Config 的配方函数
```

用 `pip install -r torchtitan/rl/examples/verifiers/my_task/requirements.txt` 安装其依赖。从 [`verifiers_dapo_math.py`](../../../../torchtitan_recipes/rl/verifiers_dapo_math.py) 的副本开始编写配方，并修改其 `VerifiersRollouter.Config`：在训练和验证数据集中填入你的 `TasksetConfig`，在 `verifiers_env_server.environment` 中填入你的 agent、harness 和运行时。然后运行：

```bash
python -m torchtitan.rl.train \
  --module torchtitan_recipes.rl.verifiers_my_task \
  --config <recipe function name>
```

完整的单轮示例见 [DAPO Math](./dapo_math)，多轮 agent 示例见 [Terminal-Bench](./terminal_bench)。本集成固定使用 Verifiers 0.4.0，并使用其 `verifiers.v1` API。