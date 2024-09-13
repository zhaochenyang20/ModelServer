# ModelServer

A ModelServer class based on the SGLang framework. Fully self-built, suggestions for further optimization are welcome. Using SGLang v0.2.15 right now.

You can also refer to the [Chinese Readme](./README_CN.md).

The ModelServer framework implements efficient, flexible, and highly fault-tolerant model service management. It can adapt to models of different scales and diverse task requirements, providing a reliable infrastructure for the deployment and application of large-scale language models.

## Get Started

Step 1: Install SGLang:
Below are the dependencies for the SGLang framework:
```bash
pip install sglang==0.2.15
pip install flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu121/torch2.3/
pip install vllm==0.5.5 # lower version will lead to errors about multimodal-config
pip install triton==2.3.1
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```
It is recommended to follow the above specified versions to avoid potential errors.

Step 2: Modify `client_config.py`:
Modify the IP address of the server in `client_config.py`:
```python
SERVER_IP = "[SECRET IP, REPLACE WITH YOURS]"
```

Step 3: Run the server
```bash
python serve_llm_pipeline.py
```

Step 4: Test the server
```bash
python model_server.py
```

## Code Structure

### `client_configs.py`

#### Constants

- **Server Configuration**: Configurations for all servers, hosting models of different sizes (e.g., `8B` and `70B`) as well as embedding models. Each server is represented by a `Server` named tuple, containing attributes such as `ip` (IP address), `port` (port number), `model_size` (model size), `model_path` (model path), and `gpus` (GPU configuration).
- **BENCHMAK_MESSAGE**: Defines a benchmark message (`BENCHMAK_MESSAGE`) used to test the performance of different servers.
- **Completion_Servers**: List of server configurations for dialogue models.
- **Embedding_Servers**: List of server configurations for embedding models (newly added).

#### Functions

- `get_fastest_server`: Tests the latency of each server and returns the fastest server along with its latency. Servers with latency higher than the current lowest latency are skipped, which is particularly significant when server latencies are highly uneven. (Support for embedding model servers has been added)
- `get_all_latency`: Checks and prints the latency of all configured servers, including both completion and embedding model servers.
- `get_running_server_sizes`: Returns a list of model sizes currently running on servers.

### `serve_llm_pipeline.py`

#### Functions

- `get_eno1_inet_address`: Retrieves the IP address associated with the `eno1` network interface.
- `is_gpu_free`: Checks if the specified GPU is free (memory usage below a certain threshold).
- `get_gpu_memory_info`: Gets total and available memory information for specified GPUs.
- `get_free_memory_ratio`: Calculates the ratio of available memory to total memory for GPUs.
- `get_comond_infos`: Dynamically constructs commands to start servers, generating appropriate startup parameters based on server configurations.
- `main`: Main function that manages GPU availability and starts model servers. Uses `ThreadPoolExecutor` to concurrently manage multiple servers.

#### Features

- **Dynamic Resource Management:** Dynamically checks and manages GPU resources, ensuring servers are only started when resources are sufficient.
- **Server Initialization:** Automatically starts servers, ensuring correct configuration and resource allocation.
- **Concurrency:** Uses `ThreadPoolExecutor` to concurrently manage multiple servers, maximizing resource utilization.
- **GPU Memory Management:** Real-time monitoring of GPU memory usage, dynamically adjusting server startup strategies.

### `model_server.py`

#### Constants

- `LATENCY_GROWING_RATE`: Latency growth rate, used for dynamically adjusting latency thresholds.
- `MAX_RETRY`: Maximum number of retry attempts, improving system fault tolerance.
- `INF`: Represents infinity, used for initializing latency comparisons.

#### `ModelServer` Class

Manages the creation and interaction of different model servers (including completion and embedding models) using automated restart and fault recovery methods.

##### Latency Management and Automatic Restart

- **Latency Monitoring:** Real-time monitoring of server response time in the `get_completion_or_embedding` method.
- **Dynamic Threshold Adjustment:** Uses `LATENCY_GROWING_RATE` to dynamically adjust acceptable latency thresholds.
- **Automatic Restart:** Triggers the `_manage_model_server` method to rebuild server connections when response times are detected to be too long.

##### Fault Tolerance Mechanism

- **Multiple Attempts:** Uses the `MAX_RETRY` mechanism to attempt requests multiple times in case of errors.
- **Error Handling:** Captures and logs exceptions, attempting to rebuild server connections.
- **Graceful Degradation:** Gracefully shuts down the service through the `turn_off_running_flag` method when all attempts fail.

##### Server Construction and Rebuilding Logic

- **Initial Construction:** Selects the fastest server based on current configurations.
- **Dynamic Rebuilding:** Automatically selects a new, faster server when server performance degrades.
- **Resource Optimization:** Balances service quality and resource utilization through reasonable rebuilding strategies.

##### New Features

- **Embedding Model Support:** Added support for embedding models, capable of handling text embedding requests.
- **Configuration File Support:** Introduced configuration file management for server running states, increasing flexibility.
- **Separate Completion and Embedding Methods:** The `get_completion_or_embedding` method handles completion and embedding tasks separately based on request type.

## Usage Suggestions

1. Set `LATENCY_GROWING_RATE` and `MAX_RETRY` reasonably to balance system response speed and stability.
2. Monitor GPU resource usage and adjust server configurations as needed.
3. Regularly check and update `BENCHMAK_MESSAGE` to ensure it effectively tests server performance.
4. Consider adding more servers or optimizing existing server configurations under high load conditions.
5. Utilize embedding model functionality for text analysis and similarity calculation tasks.


## Trouble Shooting

1. If you encounter the error `eno1` not found, you can directly remove `get_eno1_inet_address` in `serve_llm_pipeline.py` and set the IP address manually.

2. If you encounter the error `RuntimeError: Tried to instantiate class '_core_C.ScalarType', but it does not exist! Ensure that it is registered via torch::class_<ScalarType, Base, torch::detail::intrusive_ptr_target>::declare("torch._C.ScalarType");`, you can solve it by installing the correct version of torch:
```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 
```

3. If you encounter the error `ImportError: /usr/lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC_2.34' not found (required by /xxx/.triton/cache/41ce1f58e0a8aa9865e66b90d58b3307bb64c5a006830e49543444faf56202fc/cuda_utils.so)`, you can solve it by deleting the cache:
```bash
rm -rf /xxx/.triton/cache/*
```