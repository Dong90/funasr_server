import os
import json
import asyncio
import websockets
import numpy as np
import logging
import sys
from logging.handlers import RotatingFileHandler
from funasr import AutoModel

# 配置日志
# 创建logs目录
os.makedirs('logs', exist_ok=True)

# 设置根日志记录器
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # 设置为最低级别，捕获所有日志

# 创建控制台处理器
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.DEBUG)
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)

# 创建文件处理器
file_handler = RotatingFileHandler('logs/server.log', maxBytes=10485760, backupCount=5)
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)

# 添加处理器到根记录器
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# 设置该模块的日志记录器
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# FunASR库的日志记录
funasr_logger = logging.getLogger('funasr')
funasr_logger.setLevel(logging.DEBUG)

# websockets库的日志记录
websockets_logger = logging.getLogger('websockets')
websockets_logger.setLevel(logging.DEBUG)

# 设置第三方库的日志级别
logging.getLogger('asyncio').setLevel(logging.DEBUG)

logger.info("日志系统初始化完成")

class ASRServer:
    def __init__(self):
        # 加载模型
        logger.info("正在加载语音识别模型...")
        self.model = AutoModel(
            model="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"  # 标点符号恢复
        )
        logger.info("模型加载完成！")
        self.clients = {}  # 存储客户端连接

    async def process_audio(self, websocket, path):
        client_id = id(websocket)
        self.clients[client_id] = {"audio_buffer": b"", "sample_rate": 16000}
        logger.info(f"新客户端连接: {client_id}")

        try:
            async for message in websocket:
                if isinstance(message, str):
                    # 处理控制消息
                    data = json.loads(message)
                    logger.debug(f"接收到字符串消息: {data}")
                    if data.get("type") == "config":
                        self.clients[client_id]["sample_rate"] = data.get("sample_rate", 16000)
                        logger.info(f"客户端 {client_id} 配置更新: sample_rate={self.clients[client_id]['sample_rate']}")
                    
                    elif data.get("type") == "eof":
                        # 处理完整音频段
                        logger.debug(f"接收到EOF标记，处理完整音频段，音频大小: {len(self.clients[client_id]['audio_buffer'])} 字节")
                        result = await self.recognize_audio(client_id)
                        await websocket.send(json.dumps(result))
                        self.clients[client_id]["audio_buffer"] = b""
                
                elif isinstance(message, bytes):
                    # 累积音频数据
                    logger.debug(f"接收到音频数据: {len(message)} 字节")
                    self.clients[client_id]["audio_buffer"] += message
                    
                    # 当缓冲区大于一定大小时进行识别
                    if len(self.clients[client_id]["audio_buffer"]) > 32000:  # 约2秒@16kHz
                        logger.debug(f"缓冲区达到阈值，进行识别，缓冲区大小: {len(self.clients[client_id]['audio_buffer'])} 字节")
                        result = await self.recognize_audio(client_id)
                        await websocket.send(json.dumps(result))
                        self.clients[client_id]["audio_buffer"] = b""
        
        except websockets.exceptions.ConnectionClosed as e:
            logger.info(f"客户端断开连接: {client_id}, 原因: {str(e)}")
        except Exception as e:
            logger.error(f"处理客户端 {client_id} 数据时出错: {str(e)}", exc_info=True)
        finally:
            if client_id in self.clients:
                del self.clients[client_id]
                logger.info(f"客户端 {client_id} 资源已清理")

    async def recognize_audio(self, client_id):
        client_data = self.clients[client_id]
        audio_data = client_data["audio_buffer"]
        sample_rate = client_data["sample_rate"]
        
        if not audio_data:
            logger.warning(f"客户端 {client_id} 的音频数据为空")
            return {"text": "", "timestamps": []}
        
        # 将字节数据转换为numpy数组
        logger.debug(f"准备处理音频数据，大小: {len(audio_data)} 字节, 采样率: {sample_rate}")
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0
        logger.debug(f"音频数据转换为numpy数组，形状: {audio_np.shape}")
        
        try:
            # 执行语音识别，获取文本和时间戳
            logger.info(f"开始识别客户端 {client_id} 的音频数据")
            result = self.model.generate(
                input=audio_np,
                batch_size_s=8,  # 批处理大小
                hotword=None,    # 热词支持
                param_dict={
                    "audio_sample_rate": sample_rate,
                    "output_word_timestamp": True  # 启用时间戳输出
                }
            )
            logger.debug(f"识别结果类型: {type(result)}, 内容: {result}")
            
            # 确保result是字典类型
            if not isinstance(result, dict):
                logger.error(f"模型返回了非字典格式的结果: {result}")
                return {"text": "", "error": "模型返回格式错误", "timestamps": []}
            
            # 处理识别结果
            text = result.get("text", "")
            timestamps = []
            
            # 处理时间戳信息
            if "timestamp" in result:
                timestamp_data = result["timestamp"]
                logger.debug(f"处理时间戳信息类型: {type(timestamp_data)}, 内容: {timestamp_data}")
                
                # 处理时间戳可能是列表的情况
                if isinstance(timestamp_data, list):
                    for segment in timestamp_data:
                        try:
                            if isinstance(segment, dict) and "text" in segment and "timestamp" in segment:
                                timestamps.append({
                                    "text": segment["text"],
                                    "start": segment["timestamp"][0],
                                    "end": segment["timestamp"][1]
                                })
                            else:
                                logger.warning(f"时间戳分段格式异常: {segment}")
                        except Exception as ts_error:
                            logger.error(f"处理时间戳分段时出错: {str(ts_error)}")
                else:
                    logger.warning(f"时间戳数据不是列表格式: {timestamp_data}")
            
            logger.info(f"客户端 {client_id} 的音频识别完成，文本长度: {len(text)}")
            return {
                "text": text,
                "timestamps": timestamps
            }
        
        except Exception as e:
            logger.error(f"识别客户端 {client_id} 的音频时出错: {str(e)}", exc_info=True)
            return {"text": "", "error": str(e), "timestamps": []}

    async def start_server(self, host='0.0.0.0', port=8080):
        logger.info(f"正在启动WebSocket服务器，监听地址: {host}:{port}")
        server = await websockets.serve(self.process_audio, host, port)
        logger.info(f"服务器启动成功，监听地址: {host}:{port}")
        return server

async def main():
    logger.info("服务器主程序开始启动")
    asr_server = ASRServer()
    server = await asr_server.start_server(host="127.0.0.1", port=8081)
    logger.info("服务器已启动，等待连接...")
    await asyncio.Future()  # 运行直到被取消

if __name__ == "__main__":
    logger.info("程序入口点，开始执行主程序")
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("收到键盘中断，程序退出")
    except Exception as e:
        logger.critical(f"程序异常退出: {str(e)}", exc_info=True)
    finally:
        logger.info("程序退出") 