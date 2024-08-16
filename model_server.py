"""
model_server.py
"""

import os
import time
from typing import Dict, List
import openai
import random
from client_configs import (
    get_fastest_server,
    get_running_server_sizes,
    MODEL_NAME_8B,
    MODEL_NAME_70B,
    BENCHMAK_MESSAGE,
)
from IPython import embed
import random
import math

os.environ["PYTHONUTF8"] = "1"

LATENCY_GROWING_RATE = 20
MAX_RETRY = 5
INF = 100


class ModelServer:
    def __init__(self) -> None:
        running_server_sizes = get_running_server_sizes()
        self.client_70b, self.client_8b = None, None
        self.latency_70b, self.latency_8b = INF, INF
        if "70" in running_server_sizes:
            self._manage_model_server(latency_bound=3, model_size="70")
        if "8" in running_server_sizes:
            self._manage_model_server(latency_bound=3, model_size="8")

    def _manage_model_server(self, latency_bound, model_size: str) -> None:
        build_latency = latency_bound
        build_count = 0
        status = False
        while not status:
            server, latency_bound = get_fastest_server(
                initial_latency=build_latency, model_size=model_size
            )
            if server is not None:
                client = openai.OpenAI(
                    base_url=f"http://{server.ip}:{server.port}/v1",
                    api_key="EMPTY",
                )
                if model_size == "70":
                    self.client_70b, self.latency_70b = client, latency_bound
                else:
                    self.client_8b, self.latency_8b = client, latency_bound
                print(
                    f"Model server {model_size}B built with latency_bound {latency_bound}."
                )
                status = True
            else:
                build_latency *= LATENCY_GROWING_RATE
                build_count += 1
                print(
                    f"Attempt {build_count} to build model server {model_size}B failed."
                )
                if build_count > MAX_RETRY:
                    raise RuntimeError(
                        f"Could not build model server after {MAX_RETRY} attempts."
                    )

    def get_completion(
        self,
        model_size: str,
        message: List,
        temperature: float = 0.0,
        max_tokens: int = 256,
    ) -> str:
        # print(f"Message: {message}")
        model_name = MODEL_NAME_70B if model_size == "70" else MODEL_NAME_8B

        for attempt in range(MAX_RETRY):
            try:
                assert (self.client_70b is not None and model_size == "70") or (
                    self.client_8b is not None and model_size == "8"
                ), "Model server not initialized."
                client = self.client_70b if model_size == "70" else self.client_8b
                latency_bound = (
                    self.latency_70b if model_size == "70" else self.latency_8b
                )
                print(
                    f"Using client {client.base_url} with latency bound {latency_bound}."
                )
                start_time = time.time()
                completion = client.chat.completions.create(
                    model=model_name,
                    messages=message,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    stop=["<|eot_id|>"],
                )
                elapsed_time = time.time() - start_time
                print(f"Connection Time: {elapsed_time:.3f} s")

                if elapsed_time >= LATENCY_GROWING_RATE * latency_bound:
                    print(
                        f"Rebuilding model seed due to response delay ({elapsed_time:.3f}) longer than {LATENCY_GROWING_RATE} * latency bound ({latency_bound:.3f})."
                    )
                    self._manage_model_server(
                        latency_bound=LATENCY_GROWING_RATE * latency_bound,
                        model_size=model_size,
                    )

                return str(completion.choices[0].message.content)

            except Exception as e:
                print(f"Attempt {attempt + 1} to get response failed with error: {e}")
                print(f"Rebuilding model server {model_size}B.")
                self._manage_model_server(
                    latency_bound=INF,
                    model_size=model_size,
                )

        raise RuntimeError(
            f"All clients failed to produce a completion after {MAX_RETRY} attempts."
        )


if __name__ == "__main__":
    #! Test the model server
    server = ModelServer()
    message = BENCHMAK_MESSAGE
    # message = []
    #! 这个 message 用于测试重启功能
    for i in range(10):
        print(f"Completion {i}:")
        complition = server.get_completion("8", message)
        print(complition)
