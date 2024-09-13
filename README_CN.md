# ModelServer

基于 SGLang 框架的 ModelServer 类。完全自建，还请提出更多优化方式。采用的 SGLang 版本为 v0.2.15。

ModelServer 框架实现了高效、灵活且具有强大容错能力的模型服务管理。它能够适应不同规模的模型和多样的任务需求，为大规模语言模型的部署和应用提供了可靠的基础设施。

## 快速使用

### 安装 SGLang

参考我当前的配置，安装 SGLang 和依赖项。

```bash
pip install sglang==0.2.15
pip install flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu121/torch2.3/

# 较低版本的vllm可能导致关于multimodal-config的错误
pip install vllm==0.5.5

pip install triton==2.3.1

# 根据您的本地设备更改cuda版本
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```

### 修改 `client_config.py`

在 `client_config.py` 中配置你本地的服务器的 IP 地址和模型路径等等：

```python
SERVER_IP = "[SECRET IP, REPLACE WITH YOURS]"
MODEL_NAME_8B = "8bins"
MODEL_NAME_70B = "70bins"
EMBEDDING_7B = "7embed"
```

### 启动 Model Engine

```bash
python serve_llm_pipeline.py
```

### 测试服务器延迟

```python
python client_config.py
```

### 测试 ModelServer

```bash
python model_server.py
```

## 代码结构

### `client_configs.py`

#### 常量

- **Server 配置**：全部服务器配置，托管了不同大小的模型（如 `8B` 和 `70B`）以及嵌入模型。每个服务器通过一个 `Server` 命名元组来表示，包含属性如 `ip`（IP 地址）、`port`（端口）、`model_size`（模型大小）、`model_path`（模型路径）和 `gpus`（GPU 配置）。
- **BENCHMAK_MESSAGE**： 定义了一个基准消息（`BENCHMAK_MESSAGE`），用于测试不同服务器的性能。
- **Completion_Servers**：对话模型的服务器配置列表。
- **Embedding_Servers**：嵌入模型的服务器配置列表（新增）。

#### 函数

- `get_fastest_server`： 测试各个服务器的延迟，并返回最快的服务器及其延迟。延迟长于当前最低 latency 的服务器将被跳过，这在服务器 latency 非常不均衡时有显著意义。（新增了对嵌入模型服务器的支持）
- `get_all_latency`： 检查并打印所有配置服务器的延迟，包括完成模型和嵌入模型服务器。
- `get_running_server_sizes`： 返回当前运行中的服务器模型大小列表。

### `serve_llm_pipeline.py`

#### 函数

- `get_eno1_inet_address`： 获取网络接口 `eno1` 相关的 IP 地址。
- `is_gpu_free`： 检查指定的 GPU 是否空闲（内存使用量低于某个阈值）。
- `get_gpu_memory_info`： 获取指定 GPU 的总内存和可用内存信息。
- `get_free_memory_ratio`： 计算 GPU 可用内存与总内存的比例。
- `get_comond_infos`： 动态构建用于启动服务器的命令，根据服务器的配置生成适当的启动参数。
- `main`： 主函数，管理 GPU 的可用性并启动模型服务器。使用 `ThreadPoolExecutor` 实现并发管理多个服务器。

#### features

- **动态资源管理：** 动态检查和管理 GPU 资源，确保只有在资源充足时才启动服务器。
- **服务器初始化：** 自动化地启动服务器，确保使用正确的配置和资源分配。
- **并发性：** 使用 `ThreadPoolExecutor` 并发地管理多个服务器，充分利用资源。
- **GPU 内存管理：** 实时监控 GPU 内存使用情况，动态调整服务器启动策略。

### `model_server.py`

#### 常量

- `LATENCY_GROWING_RATE`：延迟增长率，用于动态调整延迟阈值。
- `MAX_RETRY`：最大重试次数，提高系统容错能力。
- `INF`：表示无穷大，用于初始化延迟比较。

####  `ModelServer` 类

采用自动化重启和容错恢复的方式管理不同模型服务器（包括完成模型和嵌入模型）的创建和交互。

##### 延迟管理和自动重启

- **延迟监测：** 在 `get_completion_or_embedding` 方法中实时监测服务器响应时间。
- **动态阈值调整：** 使用 `LATENCY_GROWING_RATE` 动态调整可接受的延迟阈值。
- **自动重启：** 当检测到响应时间过长时，触发 `_manage_model_server` 方法重建服务器连接。

##### 容错机制

- **多次尝试：** 使用 `MAX_RETRY` 机制，在发生错误时多次尝试请求。
- **错误处理：** 捕获并记录异常，尝试重建服务器连接。
- **优雅降级：** 当所有尝试失败时，通过 `turn_off_running_flag` 方法优雅地关闭服务。

##### 服务器构建与重建逻辑

- **初始化构建：** 根据当前配置选择最快的服务器。
- **动态重建：** 当服务器性能下降时，自动选择新的更快服务器。
- **资源优化：** 通过合理的重建策略，平衡服务质量和资源利用。

##### 新特性

- **嵌入模型支持：** 新增对嵌入模型的支持，可以处理文本嵌入请求。
- **配置文件支持：** 引入配置文件管理服务器运行状态，提高灵活性。
- **分离的完成和嵌入方法：** `get_completion_or_embedding` 方法根据请求类型分别处理完成和嵌入任务。

## 使用建议

1. 合理设置 `LATENCY_GROWING_RATE` 和 `MAX_RETRY`，以平衡系统响应速度和稳定性。
2. 监控 GPU 资源使用情况，适时调整服务器配置。
3. 定期检查和更新 `BENCHMAK_MESSAGE`，确保其能够有效测试服务器性能。
4. 在高负载情况下，考虑增加更多的服务器或优化现有服务器配置。
5. 利用嵌入模型功能进行文本分析和相似度计算任务。

## Trouble Shooting

1. 如果遇到`eno1`未找到的错误，可以直接在`serve_llm_pipeline.py`中移除`get_eno1_inet_address`并手动设置IP地址。（为了在具有不同 IP 的多个集群上运行 Model Engine 时，我采用 IP 地址来区分集群。如果不需要在多个集群上运行，就不用区分。）
2. 如果遇到以下错误：

```bash
RuntimeError: Tried to instantiate class '_core_C.ScalarType', but it does not exist! Ensure that it is registered via torch::class_<ScalarType, Base, torch::detail::intrusive_ptr_target>::declare("torch._C.ScalarType");
```

可以通过安装正确版本的torch来解决：

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121
```

1. 如果遇到以下错误：

```bash
ImportError: /usr/lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC_2.34' not found (required by /xxx/.triton/cache/41ce1f58e0a8aa9865e66b90d58b3307bb64c5a006830e49543444faf56202fc/cuda_utils.so)
```

可以通过删除缓存来解决：

```bash
rm -rf /xxx/.triton/cache/*
```
