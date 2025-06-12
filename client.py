import asyncio
import json
import websockets
import pyaudio
import numpy as np
import time
import argparse
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ASRClient:
    def __init__(self, server_url="ws://localhost:8080"):
        self.server_url = server_url
        self.sample_rate = 16000  # 采样率
        self.chunk_size = 1600    # 每帧大小 (100ms @ 16kHz)
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.is_recording = False
        self.websocket = None
        self.loop = None  # 存储主事件循环的引用
        
        # 用于存储识别结果
        self.current_text = ""         # 当前段的文本
        self.accumulated_text = ""     # 累积的所有文本
        self.timestamps = []           # 当前段的时间戳
        self.session_start_time = 0    # 会话开始时间
    
    async def connect(self):
        """连接到ASR服务器"""
        try:
            self.loop = asyncio.get_running_loop()
            self.websocket = await websockets.connect(self.server_url)
            await self.websocket.send(json.dumps({
                "type": "config",
                "sample_rate": self.sample_rate
            }))
            logger.info(f"已连接到服务器: {self.server_url}")
            return True
        except Exception as e:
            logger.error(f"连接服务器失败: {str(e)}")
            return False
    
    def start_recording(self):
        """开始录音"""
        if self.is_recording:
            return
        
        self.session_start_time = time.time()
        self.accumulated_text = ""
        
        def audio_callback(in_data, frame_count, time_info, status):
            if self.websocket is not None and self.loop is not None:
                try:
                    asyncio.run_coroutine_threadsafe(
                        self.websocket.send(in_data),
                        self.loop
                    )
                except Exception as e:
                    logger.error(f"发送音频数据失败: {str(e)}")
            return (in_data, pyaudio.paContinue)
        
        self.stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=audio_callback
        )
        
        self.is_recording = True
        logger.info("开始录音")
        print("\n【录音已开始】实时识别已启动，您可以开始说话...")
    
    def stop_recording(self):
        """停止录音"""
        if not self.is_recording:
            return
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
        
        self.is_recording = False
        logger.info("停止录音")
        
        if self.websocket is not None and self.loop is not None:
            try:
                asyncio.run_coroutine_threadsafe(
                    self.websocket.send(json.dumps({"type": "eof"})),
                    self.loop
                )
            except Exception as e:
                logger.error(f"发送EOF标记失败: {str(e)}")
    
    async def receive_results(self):
        """接收并处理识别结果"""
        if not self.websocket:
            return
        
        try:
            while True:
                message = await self.websocket.recv()
                try:
                    result = json.loads(message)
                    
                    # 处理错误响应
                    if isinstance(result, dict) and "error" in result:
                        logger.error(f"服务器返回错误: {result['error']}")
                        continue
                    
                    # 提取文本
                    text = ""
                    if isinstance(result, dict) and "text" in result:
                        text = result["text"]
                        if text:
                            self.current_text = text
                            if not self.accumulated_text.endswith(text) and text not in self.accumulated_text:
                                if self.accumulated_text:
                                    self.accumulated_text += " " + text
                                else:
                                    self.accumulated_text = text
                    
                    # 显示结果
                    self._display_results(text)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"JSON解析错误: {str(e)}")
                except Exception as e:
                    logger.error(f"处理结果时出错: {str(e)}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("服务器连接已关闭")
        except Exception as e:
            logger.error(f"接收结果时出错: {str(e)}")
    
    def _display_results(self, text):
        """显示识别结果"""
        if not text:
            return
        
        print("\033c", end="")  # 清屏
        
        print("\n===== 实时语音识别结果 =====")
        
        # 显示当前识别的文本
        if text:
            print(f"\n【当前识别】: {text}")
        
        # 显示累积的文本
        if self.accumulated_text:
            print(f"\n【累积文本】: {self.accumulated_text}")
        
        # 显示会话时长
        if self.session_start_time > 0:
            session_duration = time.time() - self.session_start_time
            print(f"\n【会话时长】: {session_duration:.1f}秒")
        
        print("\n提示: 按 's' 停止/开始录音, 'q' 退出程序")
        print("=====================")
    
    async def disconnect(self):
        """断开连接"""
        if self.websocket:
            await self.websocket.close()
            self.websocket = None
        
        if self.audio:
            self.audio.terminate()
            self.audio = None
        
        logger.info("已断开连接")

async def main():
    parser = argparse.ArgumentParser(description='FunASR实时语音识别客户端')
    parser.add_argument('--server', type=str, default='ws://127.0.0.1:8081',
                        help='ASR服务器WebSocket URL')
    args = parser.parse_args()
    
    client = ASRClient(server_url=args.server)
    
    if not await client.connect():
        return
    
    asyncio.create_task(client.receive_results())
    
    try:
        print("\n=== FunASR实时语音识别 ===")
        print("按 's' 开始/停止录音")
        print("按 'q' 退出程序\n")
        
        client.start_recording()  # 默认开始录音
        
        while True:
            cmd = await asyncio.get_event_loop().run_in_executor(None, input, "输入命令> ")
            
            if cmd.lower() == 'q':
                break
            elif cmd.lower() == 's':
                if client.is_recording:
                    client.stop_recording()
                else:
                    client.start_recording()
    
    except KeyboardInterrupt:
        pass
    finally:
        client.stop_recording()
        await client.disconnect()

if __name__ == "__main__":
    asyncio.run(main()) 