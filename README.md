# ModelServer

A ModelServer class based on the SGLang framework. Fully self-built, suggestions for further optimization are welcome. Using SGLang v0.2.15 right now.

You can also refer to the [Chinese Readme](./README_CN.md).

The ModelServer framework implements efficient, flexible, and highly fault-tolerant model service management. It can adapt to models of different scales and diverse task requirements, providing a reliable infrastructure for the deployment and application of large-scale language models.

After configuration, you can get a response or embedding from these kind of commands:

```
from model_sever import ModelServer
message = [
    {
        "role": "user",
        "content": "Interact with a household to solve a task. Here are a few examples.\nYou are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 1, a diningtable 3, a diningtable 2, a diningtable 1, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sidetable 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.\nYour task is to: find some apple and put it in sidetable.\n> think: To solve the task, I need to find and take an apple, then put it in sidetable.\nOK.\n> think: First I need to find an apple. An apple is more likely to appear in fridges (1), diningtables (1-3), sidetables (1), countertops (1), sinkbasins (1), garbagecan (1). I can check one by one, starting with fridge 1.\nOK.\n> go to fridge 1\nThe fridge 1 is closed.\n> open fridge 1\nYou open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, a potato 2, and a tomato 1.\n> go to diningtable 1\nOn the diningtable 1, you see nothing.\n> go to diningtable 2\nOn the diningtable 2, you see a pineapple 1.\n> go to diningtable 3\nOn the diningtable 3, you see a bowl 2, a bowl 1, a bread 2, and a bread 1.\n> go to sidetable 1\nOn the sidetable 1, you see a cup 1, a lettuce 1, a peppershaker 3, a potato 1, and a saltshaker 1.\n> go to countertop 1\nOn the countertop 1, you see a butterknife 1, a dishsponge 1, and a soapbottle 1.\n> go to sinkbasin 1\nOn the sinkbasin 1, you see a cup 2, a egg 1, a mug 1, and a potato 3.\n> go to garbagecan 1\nOn the garbagecan 1, you see a apple 3, and a egg 3.\n> think: Now I find an apple (3). Next, I need to take it.\n> take apple 3 from garbagecan 1\nYou pick up the apple 3 from the garbagecan 1.\n> think: Now I find an apple (3). Next, I need to put it in/on sidetable 1.\nOK.\n> go to sidetable 1\nOn the sidetable 1, you see a cup 1, a lettuce 1, a peppershaker 3, a potato 1, and a saltshaker 1.\n> put apple 3 in/on sidetable 1\nYou put the apple 3 in/on the sidetable 1.\nYou are in the middle of a room. Looking quickly around you, you see a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a countertop 1, a garbagecan 1, a handtowelholder 2, a handtowelholder 1, a sinkbasin 2, a sinkbasin 1, a toilet 1, a toiletpaperhanger 1, and a towelholder 1.\nYour task is to: put some spraybottle on toilet.\n> think: To solve the task, I need to find and take a sparybottle, then put it on toilet.\nOK.\n> think: First I need to find a spraybottle. A spraybottle is more likely to appear in cabinet (1-4), countertop (1), toilet (1), sinkbasin (1-2), garbagecan (1). I can check one by one, starting with cabinet 1.\nOK.\n> go to cabinet 1\nOn the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1.\n> go to cabinet 2\nThe cabinet 2 is closed.\n> open cabinet 2\nYou open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2.\n> think: Now I find a spraybottle (2). Next, I need to take it.\n> take spraybottle 2 from cabinet 2\nYou pick up the spraybottle 2 from the cabinet 2.\n> think: Now I take a spraybottle (2). Next, I need to put it in/on toilet 1.\nOK.\n> go to toilet 1\nOn the toilet 1, you see a soapbottle 2.\n> put spraybottle 2 in/on toilet 1\nYou put the spraybottle 2 in/on the toilet 1.\n\n\nHere is the task:\nYou are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.\nYour task is to: put a mug in desk.\n>",
    },
    {
        "role": "assistant",
        "content": "think: To solve the task, I need to find and take a mug, then put it in/on desk.",
    },
    {"role": "user", "content": "OK."},
    {
        "role": "assistant",
        "content": "think: First I need to find a mug. A mug is more likely to appear in desk (1-2), drawer (1-6), shelf (1-6), garbagecan (1), laundryhamper (1), safe (1). I can check one by one, starting with desk 1.",
    },
    {"role": "user", "content": "OK."},
    {"role": "assistant", "content": "go to desk 1"},
    {
        "role": "user",
        "content": "On the desk 1, you see a bowl 1, a cd 2, a creditcard 2, a desklamp 1, a keychain 1, a laptop 2, a pen 2, and a pencil 2.",
    },
    {
        "role": "assistant",
        "content": "think: Now I don't find a mug in desk 1. Next, I need to check desk 2.",
    },
    {"role": "user", "content": "OK."},
    {"role": "assistant", "content": "go to desk 2"},
    {
        "role": "user",
        "content": "On the desk 2, you see a alarmclock 1, a book 3, a bowl 2, a creditcard 3, and a keychain 3.",
    },
]

server = ModelServer()
# `8` here is the completion model size configured in the `client_configs`
complition = server.get_completion_or_embedding("8", message)
# get the completion from a 8b instruct model, i.e. Llama3.1 8B
print(complition)

# `7` here is the embedding model size configured in the `client_configs`
embedding = server.get_completion_or_embedding(
    "7",
    message="As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
    get_embedding=True,
)
# get the embedding of a 7b embedding model, i.e. `Alibaba-NLP/gte-Qwen1.5-7B-instruct`
print(embedding[:10])
```

## Get Started

### Install SGLang

Below are the dependencies for the SGLang framework in my framework currently. Will update later.

```bash
pip install sglang==0.2.15
pip install flashinfer==0.1.6 -i https://flashinfer.ai/whl/cu121/torch2.3/

# lower version of vllm will lead to errors about multimodal-config
pip install vllm==0.5.5

pip install triton==2.3.1

# change the cuda version according to your local device
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121
```
It is recommended to follow the above specified versions to avoid potential errors.

### Modify `client_config.py`

Modify the IP address of the server and the model path in `client_config.py`:

```python
SERVER_IP = "[SECRET IP, REPLACE WITH YOURS]"
MODEL_NAME_8B = "8bins"
MODEL_NAME_70B = "70bins"
EMBEDDING_7B = "7embed"
```

### Run the Server Engine

```bash
python serve_llm_pipeline.py
```

### Test the Server Latency

```bash
python client_config.py
```

### Test the ModelServer
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

1. If you encounter the error `eno1` not found, you can directly remove `get_eno1_inet_address` in `serve_llm_pipeline.py` and set the IP address manually. IP address is used to differentiate clusters if you want to run engines on multiple clusters with different IPs. If you only have one cluster/node, you can manually set its IP address.

2. If you encounter the error:

```
RuntimeError: Tried to instantiate class '_core_C.ScalarType', but it does not exist! Ensure that it is registered via torch::class_<ScalarType, Base, torch::detail::intrusive_ptr_target>::declare("torch._C.ScalarType");
```

You can solve it by installing the correct version of torch:

```bash
pip install torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cu121 
```

3. If you encounter the error:

```
ImportError: /usr/lib/x86_64-linux-gnu/libc.so.6: version 'GLIBC_2.34' not found (required by /xxx/.triton/cache/41ce1f58e0a8aa9865e66b90d58b3307bb64c5a006830e49543444faf56202fc/cuda_utils.so)
```

You can solve it by deleting the cache:

```bash
rm -rf /xxx/.triton/cache/*
```