# FunASR实时语音识别系统

这是一个基于FunASR框架的实时语音识别系统，可以支持实时语音识别并输出句子对应的时间戳。该系统使用DAMO语音识别模型（Paraformer）进行语音识别。

## 功能特性

- 实时语音识别
- 输出句子对应时间戳
- 使用DAMO Paraformer大模型

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

1. 运行实时识别服务器：

```bash
python server.py
```

2. 运行客户端进行语音采集和识别：

```bash
python client.py
```

3. 批处理音频文件：

```bash
python file_process.py -i <音频文件或目录> -o <输出目录>
``` 