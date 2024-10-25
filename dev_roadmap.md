# 基于 SGLang 的 Model Management System

Manager 是 engine 的封装，负责管理 engine 的配置、启动、停止、检测等。

Manager 向下通过 router 实现多个 DP engine 的 load balance，向上通过基于 fastAPI 再封装一层 OpenAI Compatible API。

## Engine Controller

提供 UI 让用户编写 engine 的配置文件：
    - 模型名称/路径
    - 模型类型（completion/embedding）
    - 模型部署参数

完成配置后，点击 launch 按钮，Manager 会根据配置文件启动所有 engine。

## Engine Monitor

提供 UI 让用户查看当前所有 engine 的运行状态，类似于一个 Grafana 的 Dashboard。

- GPU 效率
- 显存占用
- Engine 参数（decode/prefill/throughput）

## Engine Router

接入 Byran/Yichuan 的 router 实现负载均衡。

## FastAPI Server

基于 fastAPI 再封装一层 OpenAI Compatible API，并且搭建一个 ChatBot UI。