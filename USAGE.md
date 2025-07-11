# FunASR实时语音识别系统使用指南

该系统基于FunASR框架，支持实时语音识别，并可输出句子对应的时间戳信息。系统使用SenseVoiceSmall模型进行语音识别处理。

## 安装依赖

在开始使用之前，请安装所需的依赖库：

```bash
pip install -r requirements.txt
```

## 服务器与客户端模式

系统分为服务器和客户端两个组件，分别由`server.py`和`client.py`实现。

### 启动服务器

```bash
python server.py
```

服务器默认监听在`0.0.0.0:8080`地址上。首次启动时会自动下载并加载SenseVoiceSmall模型。

### 启动客户端

```bash
python client.py [--server ws://localhost:8080]
```

参数说明：
- `--server`: 指定服务器WebSocket地址，默认为`ws://localhost:8080`

客户端启动后会自动连接到服务器并开始捕获麦克风音频。在客户端运行时：
- 按 `s` 键可以开始/停止录音
- 按 `w` 键可以保存当前的识别结果
- 按 `q` 键可以退出程序

## 音频文件批处理

系统还提供了用于处理本地音频文件的批处理脚本：

```bash
python file_process.py -i <输入文件或目录> -o <输出目录>
```

参数说明：
- `-i`, `--input`: 指定要处理的音频文件或包含音频文件的目录
- `-o`, `--output`: 指定识别结果的输出目录，默认为`results`

支持处理的音频格式包括：WAV、MP3、FLAC。

## 识别结果格式

识别结果以JSON格式输出，包含以下信息：

```json
{
  "text": "完整识别文本",
  "timestamps": [
    {
      "text": "句子片段1",
      "start": 0.0,
      "end": 2.5
    },
    {
      "text": "句子片段2",
      "start": 2.5,
      "end": 5.0
    }
  ]
}
```

## 高级应用

### 时间戳生成

系统会为每个句子片段生成精确的起止时间戳，便于进行音视频对齐或字幕生成等应用。

### 性能优化

在处理长音频时，可以通过以下方式优化性能：

1. 调整服务器端的`batch_size_s`参数（默认为8）
2. 调整客户端的音频缓冲区大小`chunk_size`（默认为1600，即100ms@16kHz）

## 故障排除

### 连接问题

如果客户端无法连接到服务器，请检查：
- 服务器是否正常运行
- 防火墙设置是否允许WebSocket连接
- 网络连接是否正常

### 识别质量问题

如果识别结果质量不佳：
- 确保使用良好质量的麦克风
- 减少环境噪音
- 检查音频采样率是否为16kHz

### 模型加载问题

如果模型加载失败：
- 检查网络连接是否正常
- 确保有足够的磁盘空间存储模型
- 尝试手动下载模型并放置在正确的位置 