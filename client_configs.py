"""
client_configs.py
"""

from collections import namedtuple
import openai
from typing import List, Optional
from IPython import embed
import time
import multiprocessing
import math
from typing import List, Optional

SERVER_IP = "[SECRET IP]"
MODEL_NAME_8B = "8bins"
MODEL_NAME_70B = "70bins"
INF = 100

Server = namedtuple("Server", ["ip", "port", "model_size", "model_path", "gpus"])
BENCHMAK_MESSAGE = [
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


Servers = [
    Server(
        ip=SERVER_IP,
        port=8048,
        model_size="8",
        model_path=MODEL_NAME_8B,
        gpus=[0],
    ),
    Server(
        ip=SERVER_IP,
        port=8056,
        model_size="8",
        model_path=MODEL_NAME_8B,
        gpus=[1],
    ),
    Server(
        ip=SERVER_IP,
        port=8064,
        model_size="8",
        model_path=MODEL_NAME_8B,
        gpus=[2],
    ),
    Server(
        ip=SERVER_IP,
        port=8072,
        model_size="8",
        model_path=MODEL_NAME_8B,
        gpus=[3],
    ),
    Server(
        ip=SERVER_IP,
        port=8080,
        model_size="8",
        model_path=MODEL_NAME_8B,
        gpus=[4],
    ),
    Server(
        ip=SERVER_IP,
        port=8088,
        model_size="8",
        model_path=MODEL_NAME_8B,
        gpus=[5],
    ),
    Server(
        ip=SERVER_IP,
        port=8096,
        model_size="8",
        model_path=MODEL_NAME_8B,
        gpus=[6],
    ),
    Server(
        ip=SERVER_IP,
        port=8104,
        model_size="8",
        model_path=MODEL_NAME_8B,
        gpus=[7],
    ),
    #! 以下是 70B model 的 config，请不要同时开启
    #     Server(
    #     ip=SERVER_IP,
    #     port=8400,
    #     model_size="70",
    #     model_path=MODEL_NAME_70B,
    #     gpus=[0, 1, 2, 3],
    # ),
    #     Server(
    #     ip=SERVER_IP,
    #     port=8470,
    #     model_size="70",
    #     model_path=MODEL_NAME_70B,
    #     gpus=[4, 5, 6, 7],
    # ),
]


def get_fastest_server(SERVERS=Servers, initial_latency=10, model_size="8"):
    min_latency = initial_latency
    fastest_server = None

    def test_server(server: Server):
        def get_completion(
            client,
            message: List,
            temperature: float = 0.0,
            max_tokens: int = 256,
            model_name: Optional[str] = None,
        ) -> str:
            def target(queue):
                try:
                    completion = client.chat.completions.create(
                        model=model_name,
                        messages=message,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=["<|eot_id|>"],
                    )
                    queue.put(completion)
                except Exception as e:
                    queue.put(e)

            start_time = time.time()
            queue = multiprocessing.Queue()
            process = multiprocessing.Process(target=target, args=(queue,))
            process.start()

            process.join(timeout=min_latency)  # 等待进程完成或超时

            if process.is_alive():
                print(
                    f"Timeout: server {server.ip}:{server.port} took longer than {min_latency:.3f} seconds."
                )
                process.terminate()  # 终止进程
                process.join()  # 确保进程已终止
                return None, INF
            else:
                result = queue.get()
                if isinstance(result, Exception):
                    raise result
                latency = time.time() - start_time
                print(f"Connection Time: {latency:.3f} s")
                return str(result.choices[0].message.content), latency

        client = openai.OpenAI(
            base_url=f"http://{server.ip}:{server.port}/v1",
            api_key="EMPTY",
        )

        try:
            response, latency = get_completion(
                client,
                BENCHMAK_MESSAGE,
                0.0,
                256,
                server.model_path,
            )
            if response:
                print(
                    f"""
============================================================
Cluster: {server.ip}
Port: {server.port}
Model: {server.model_path}
Size: {server.model_size}
GPUs: {server.gpus}
Latency: {latency:.3f} s
============================================================
"""
                )
                return True, latency
            else:
                return False, INF

        except Exception as e:
            print(
                f"Could not connect to server {server.ip}:{server.port} due to error: {e}"
            )
            return False, INF

    for server in SERVERS:
        if server.model_size == model_size:
            status, latency = test_server(server)
            if status and (latency < min_latency):
                min_latency = latency
                fastest_server = server

    if fastest_server:
        print(
            f"Fastest server is {fastest_server.ip}:{fastest_server.port} with latency {min_latency:.3f} s"
        )
        return fastest_server, min_latency
    else:
        print("No servers responded in a timely manner.")
        return None, INF


def get_all_latency(SERVERS=Servers):
    def test_server(server: Server):
        client = openai.OpenAI(
            base_url=f"http://{server.ip}:{server.port}/v1",
            api_key="EMPTY",
        )

        try:
            start_time = time.time()
            completion = client.chat.completions.create(
                model=server.model_path,
                messages=BENCHMAK_MESSAGE,
                max_tokens=256,
                temperature=0.9,
            )
            response = str(completion.choices[0].message.content)
            duration = time.time() - start_time
            print(
                f"""
============================================================
TESTING SERVER
Cluster: {server.ip}
Port: {server.port}
Model: {server.model_path}
Size: {server.model_size}
GPUs: {server.gpus}
============================================================
            """
            )
            print(f"Connection Time: {duration:.3f}s for {server.ip}:{server.port}")
            print(f"Get response: {response}")
            return True
        except Exception as e:
            print(
                f"""
============================================================
TESTING SERVER
Cluster: {server.ip}
Port: {server.port}
Model: {server.model_path}
Size: {server.model_size}
GPUs: {server.gpus}
============================================================
            """
            )
            print(
                f"Could not connect to server {server.ip}:{server.port} due to error: {e}"
            )
            return False

    for server in SERVERS:
        test_server(server)


def get_running_server_sizes(SERVERS=Initial_Servers):
    server_sizes = [server.model_size for server in SERVERS]
    return server_sizes


if __name__ == "__main__":
    server, min_latency = get_fastest_server(initial_latency=3)
    print(server)
    get_all_latency()
