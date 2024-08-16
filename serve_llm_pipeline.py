"""
serve_llm_pipeline.py
"""

import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
import socket
from client_configs import Server, Servers

#! 如果 GPU ulitization 较低的话，开不了这么长的 context length
MAX_CONTEXT_LENGTH = 65536
CHUNKED_PREFILL_SIZE = int(MAX_CONTEXT_LENGTH / 8)


def get_eno1_inet_address():
    result = subprocess.run(["ifconfig"], capture_output=True, text=True)
    output = result.stdout
    match = re.search(r"eno1:.*?(inet\s+(\d+\.\d+\.\d+\.\d+))", output, re.DOTALL)
    if match:
        return match.group(2)
    return None


def is_gpu_free(gpu_ids):
    gpu_ids_string = ",".join(map(str, gpu_ids))
    command = f"nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader -i {gpu_ids_string}"
    output = subprocess.check_output(command, shell=True).decode("utf-8").strip()
    used_memory = [int(x) for x in output.split("\n")]
    return all(
        used < 10000 for used in used_memory
    )  # GPUs with usage below 10000 MB are considered free


def get_gpu_memory_info(gpu_ids):
    gpu_ids_string = ",".join(map(str, gpu_ids))
    # 获取总内存
    command_total = f"nvidia-smi --query-gpu=memory.total --format=csv,nounits,noheader -i {gpu_ids_string}"
    output_total = (
        subprocess.check_output(command_total, shell=True).decode("utf-8").strip()
    )
    total_memory = [int(x) for x in output_total.split("\n")]
    # 获取剩余内存
    command_free = f"nvidia-smi --query-gpu=memory.free --format=csv,nounits,noheader -i {gpu_ids_string}"
    output_free = (
        subprocess.check_output(command_free, shell=True).decode("utf-8").strip()
    )
    free_memory = [int(x) for x in output_free.split("\n")]
    return total_memory, free_memory


def get_free_memory_ratio(gpu_ids):
    total_memory, free_memory = get_gpu_memory_info(gpu_ids)
    free_memory_ratio = [free / total for free, total in zip(free_memory, total_memory)]
    return free_memory_ratio


def get_comond_infos(server: Server):
    assert len(server.gpus) > 0, "No GPUs assigned to the server."
    assert (
        server.port % int(server.model_size) == 0
    ), "Port should be divisible by model size."
    assert (
        server.model_size in server.model_path
    ), "Model size should be in the model name."
    group_gpu_string = ",".join(map(str, server.gpus))
    tensor_parallel_size = len(server.gpus)
    command = f"""
    CUDA_VISIBLE_DEVICES={group_gpu_string} python -m sglang.launch_server --enable-p2p-check --model-path {server.model_path} \
    --dtype auto --tensor-parallel-size {tensor_parallel_size} \
    --context-length {MAX_CONTEXT_LENGTH} --chunked-prefill-size {CHUNKED_PREFILL_SIZE} \
    --port {server.port} --host 0.0.0.0 """
    #! host 0.0.0.0 可以用于广播
    if server.model_size == "8":
        command += " --enable-torch-compile "
    #! 8b 模型需要开启 torch compile，70b 还没优化
    return (group_gpu_string, command, server.port, server.model_size)


def main():
    eno1_ip_address = get_eno1_inet_address()
    ServerID = int(eno1_ip_address[-1])
    assert str(ServerID) in socket.gethostname(), "ServerID should be in the hostname."

    print(
        f"""
============================================================
Cluster: {socket.gethostname()}
IP: {get_eno1_inet_address()}
============================================================
"""
    )

    command_infos = [
        get_comond_infos(server) for server in Servers if (server.ip == eno1_ip_address)
    ]

    def run_with_gpu_check(command_info):
        group_id, command, port, model_size = command_info

        while not is_gpu_free(group_id):
            print(f"Waiting for GPU(s) {group_id} to be free...")
            time.sleep(10)

        free_gpu_ration = min(get_free_memory_ratio(group_id))
        gpu_ultization = None
        if free_gpu_ration >= 0.95:
            gpu_ultization = 0.90
        elif free_gpu_ration >= 0.85:
            gpu_ultization = 0.70
        else:
            raise ValueError("GPU memory is not enough.")
        print(
            f"Serving {model_size}b model on server {ServerID} port {port} with GPUs {group_id}"
        )
        command = command + f" --mem-fraction-static {gpu_ultization}\n"
        print(command)
        subprocess.run(command, shell=True)

    with ThreadPoolExecutor(max_workers=len(command_infos)) as executor:
        futures = [
            executor.submit(run_with_gpu_check, cmd_info) for cmd_info in command_infos
        ]
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
