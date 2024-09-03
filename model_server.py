"""
model_server.py
"""

import os

os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["PYTHONUTF8"] = "1"
import time
from typing import Dict, List
import openai
import json
from client_configs import (
    get_fastest_server,
    get_running_server_sizes,
    MODEL_NAME_70B,
    MODEL_NAME_8B,
    EMBEDDING_7B,
    EMBEDDING_2B,
    BENCHMAK_MESSAGE,
)

LATENCY_GROWING_RATE = 20
MAX_RETRY = 20
INF = 200


class ModelServer:
    def __init__(self, config_path: str = None) -> None:
        running_server_sizes = get_running_server_sizes()
        (
            self.completion_client_70b,
            self.completion_client_8b,
            self.embedding_client_7b,
            self.embedding_client_2b,
        ) = (None, None, None, None)
        self.latency_70b, self.latency_8b, self.latency_7b, self.latency_2b = (
            INF,
            INF,
            INF,
            INF,
        )
        self.config_path = config_path
        # Turn the running flag in config path when the server failed to get response
        if "70" in running_server_sizes:
            self._manage_model_server(latency_bound=3, model_size="70")
        if "8" in running_server_sizes:
            self._manage_model_server(latency_bound=3, model_size="8")
        if "7" in running_server_sizes:
            self._manage_model_server(
                latency_bound=3, model_size="7", get_embedding=True
            )
        if "2" in running_server_sizes:
            self._manage_model_server(
                latency_bound=3, model_size="2", get_embedding=True
            )

    def turn_off_running_flag(self) -> None:
        with open(self.config_path, "r", encoding="utf-8") as rf:
            info_dict = json.load(rf)
            info_dict["is_running"] = False
        with open(self.config_path, "w", encoding="utf-8") as wf:
            json.dump(info_dict, wf, indent=4)

    def _manage_model_server(
        self, latency_bound, model_size: str, get_embedding: bool = False
    ) -> None:
        build_latency = latency_bound
        build_count = 0
        status = False
        while not status:
            server, latency_bound = get_fastest_server(
                initial_latency=build_latency,
                model_size=model_size,
                test_embedding_servers=get_embedding,
            )
            # latency_bound+=10
            if server is not None:
                client = openai.OpenAI(
                    base_url=(f"http://{server.ip}:{server.port}/v1"),
                    api_key=("sk-1dwqsdv4r3wef3rvefg34ef1dwRv"),
                )
                if model_size == "70" and not get_embedding:
                    self.completion_client_70b, self.latency_70b = client, latency_bound
                elif model_size == "8" and not get_embedding:
                    self.completion_client_8b, self.latency_8b = client, latency_bound
                elif model_size == "7" and get_embedding:
                    self.embedding_client_7b, self.latency_7b = client, latency_bound
                elif model_size == "2" and get_embedding:
                    self.embedding_client_2b, self.latency_2b = client, latency_bound
                else:
                    raise NotImplementedError
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
                    assert self.config_path is not None, "Config path is required."
                    self.turn_off_running_flag()
                    raise RuntimeError(
                        f"Could not build model server after {MAX_RETRY} attempts."
                    )

    def get_completion_or_embedding(
        self,
        model_size: str,
        message,
        temperature: float = 0.0,
        max_tokens: int = 256,
        get_embedding: bool = False,
    ) -> str:
        # print(f"Message: {message}")
        assert model_size in ["70", "8", "7", "2"]
        if not get_embedding:
            model_name = MODEL_NAME_70B if model_size == "70" else MODEL_NAME_8B
        else:
            model_name = EMBEDDING_7B if model_size == "7" else EMBEDDING_2B

        for attempt in range(MAX_RETRY):
            try:
                assert (
                    (self.completion_client_70b is not None and model_size == "70")
                    or (self.completion_client_8b is not None and model_size == "8")
                    or (self.embedding_client_7b is not None and model_size == "7")
                    or (self.embedding_client_2b is not None and model_size == "2")
                ), "Model server not initialized."

                if not get_embedding:
                    client = (
                        self.completion_client_70b
                        if model_size == "70"
                        else self.completion_client_8b
                    )
                    latency_bound = (
                        self.latency_70b if model_size == "70" else self.latency_8b
                    )
                else:
                    client = (
                        self.embedding_client_7b
                        if model_size == "7"
                        else self.embedding_client_2b
                    )
                    latency_bound = (
                        self.latency_7b if model_size == "7" else self.latency_2b
                    )
                # print(
                #     f"Using client {client.base_url} with latency bound {latency_bound}."
                # )
                start_time = time.time()
                if not get_embedding:
                    assert type(message) == list, "Message should be a list."
                    response = client.chat.completions.create(
                        model=model_name,
                        messages=message,
                        max_tokens=max_tokens,
                        temperature=temperature,
                        stop=["<|eot_id|>"],
                    )
                else:
                    assert type(message) == str, "Message should be a string."
                    response = client.embeddings.create(
                        model=model_name,
                        input=message,
                    )
                elapsed_time = time.time() - start_time
                # elapsed_time = 50
                # print(f"Connection Time: {elapsed_time:.3f} s")

                if elapsed_time >= LATENCY_GROWING_RATE * latency_bound:
                    print(
                        f"Rebuilding model seed due to response delay ({elapsed_time:.3f}) longer than {LATENCY_GROWING_RATE} * latency bound ({latency_bound:.3f})."
                    )
                    self._manage_model_server(
                        latency_bound=LATENCY_GROWING_RATE * latency_bound,
                        model_size=model_size,
                        get_embedding=get_embedding,
                    )

                return (
                    str(response.choices[0].message.content)
                    if not get_embedding
                    else response.data[0].embedding
                )

            except Exception as e:
                print(f"Attempt {attempt + 1} to get response failed with error: {e}")
                print(f"Rebuilding model server {model_size}B.")
                self._manage_model_server(
                    latency_bound=INF,
                    model_size=model_size,
                    get_embedding=get_embedding,
                )

        error_message = (
            f"All clients failed to produce a completion after {MAX_RETRY} attempts."
        )
        print(error_message)
        print(message)
        assert self.config_path is not None, "Config path is required."
        self.turn_off_running_flag()
        raise RuntimeError(error_message)


if __name__ == "__main__":
    #! Test the model server
    server = ModelServer()
    message = BENCHMAK_MESSAGE
    # message = []
    for i in range(10):
        print(f"Completion {i}:")
        complition = server.get_completion_or_embedding("8", message)
        print(complition)

    embedding = None
    for i in range(10):
        print(f"Embedding {i}:")
        embedding = server.get_completion_or_embedding(
            "7",
            message="As a general guideline, the CDC's average requirement of protein for women ages 19 to 70 is 46 grams per day. But, as you can see from this chart, you'll need to increase that if you're expecting or training for a marathon. Check out the chart below to see how much protein you should be eating each day.",
            get_embedding=True,
        )
        print(embedding[:10])
