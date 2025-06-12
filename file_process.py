import os
import json
import argparse
import logging
import sys
from logging.handlers import RotatingFileHandler
import numpy as np
from funasr import AutoModel
from scipy.io import wavfile

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
file_handler = RotatingFileHandler('logs/file_process.log', maxBytes=10485760, backupCount=5)
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

# scipy库的日志记录
scipy_logger = logging.getLogger('scipy')
scipy_logger.setLevel(logging.DEBUG)

logger.info("日志系统初始化完成")

def load_audio(audio_file):
    """加载音频文件"""
    try:
        logger.info(f"开始加载音频文件: {audio_file}")
        sample_rate, audio_data = wavfile.read(audio_file)
        logger.debug(f"原始音频数据类型: {audio_data.dtype}, 形状: {audio_data.shape}, 采样率: {sample_rate}Hz")
        
        # 将数据转换为float32格式并归一化
        if audio_data.dtype == np.int16:
            logger.debug("将int16音频数据转换为float32并归一化")
            audio_data = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype == np.int32:
            logger.debug("将int32音频数据转换为float32并归一化")
            audio_data = audio_data.astype(np.float32) / 2147483648.0
        elif audio_data.dtype == np.uint8:
            logger.debug("将uint8音频数据转换为float32并归一化")
            audio_data = (audio_data.astype(np.float32) - 128) / 128.0
            
        # 如果是双通道，转换为单通道
        if len(audio_data.shape) > 1:
            logger.debug(f"检测到多通道音频({audio_data.shape[1]}通道)，转换为单通道")
            audio_data = np.mean(audio_data, axis=1)
        
        logger.debug(f"转换后音频数据: 长度={len(audio_data)}, 最大值={np.max(audio_data):.4f}, 最小值={np.min(audio_data):.4f}")
        logger.info(f"音频文件加载成功: {audio_file}")
        return sample_rate, audio_data
    
    except Exception as e:
        logger.error(f"加载音频文件失败: {str(e)}", exc_info=True)
        return None, None

def process_audio_file(audio_file, output_dir=None):
    """处理单个音频文件并输出识别结果"""
    if not os.path.exists(audio_file):
        logger.error(f"文件不存在: {audio_file}")
        return None
        
    # 加载音频文件
    sample_rate, audio_data = load_audio(audio_file)
    if sample_rate is None or audio_data is None:
        logger.error(f"无法处理音频文件: {audio_file}")
        return None
        
    logger.info(f"正在处理文件: {audio_file}")
    logger.info(f"采样率: {sample_rate}Hz, 音频长度: {len(audio_data)/sample_rate:.2f}s")
    
    # 加载语音识别模型
    logger.info("正在加载语音识别模型...")
    try:
        model = AutoModel(
            model="damo/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
            vad_model="damo/speech_fsmn_vad_zh-cn-16k-common-pytorch",
            punc_model="damo/punc_ct-transformer_zh-cn-common-vocab272727-pytorch"  # 标点符号恢复
        )
        logger.info("模型加载成功")
    except Exception as e:
        logger.error(f"加载模型失败: {str(e)}", exc_info=True)
        return None
    
    # 执行语音识别
    try:
        logger.info("开始执行语音识别...")
        result = model.generate(
            input=audio_data,
            batch_size_s=8,  # 批处理大小
            hotword=None,    # 热词支持
            param_dict={
                "audio_sample_rate": sample_rate,
                "output_word_timestamp": True  # 启用时间戳输出
            }
        )
        logger.debug(f"识别原始结果: {result}")
        
        # 整理识别结果
        output_result = {
            "filename": os.path.basename(audio_file),
            "text": result.get("text", ""),
            "timestamps": []
        }
        
        logger.info(f"识别文本结果: {output_result['text']}")
        
        # 处理时间戳信息
        if "timestamp" in result:
            timestamp_count = len(result["timestamp"]) if isinstance(result["timestamp"], list) else 0
            logger.debug(f"时间戳分段数: {timestamp_count}")
            
            for segment in result["timestamp"]:
                if isinstance(segment, dict) and "text" in segment and "timestamp" in segment:
                    timestamp_entry = {
                        "text": segment["text"],
                        "start": segment["timestamp"][0],
                        "end": segment["timestamp"][1]
                    }
                    output_result["timestamps"].append(timestamp_entry)
                    logger.debug(f"时间戳分段: {timestamp_entry}")
        
        # 保存结果
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(audio_file))[0]
            output_file = os.path.join(output_dir, f"{base_name}_result.json")
            
            logger.info(f"正在保存结果到: {output_file}")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_result, f, ensure_ascii=False, indent=2)
                
            logger.info(f"结果已保存到: {output_file}")
        
        return output_result
    
    except Exception as e:
        logger.error(f"处理文件时出错: {str(e)}", exc_info=True)
        return None

def main():
    parser = argparse.ArgumentParser(description='FunASR音频文件批处理')
    parser.add_argument('-i', '--input', required=True, help='输入音频文件或目录')
    parser.add_argument('-o', '--output', default='results', help='输出结果目录')
    args = parser.parse_args()
    
    logger.info(f"开始处理，输入: {args.input}, 输出目录: {args.output}")
    
    if os.path.isdir(args.input):
        # 处理目录中的所有WAV文件
        logger.info(f"检测到输入是目录: {args.input}，将处理所有音频文件")
        count = 0
        file_count = 0
        for root, _, files in os.walk(args.input):
            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac')):
                    file_count += 1
                    file_path = os.path.join(root, file)
                    logger.info(f"发现音频文件({file_count}): {file_path}")
                    result = process_audio_file(file_path, args.output)
                    if result:
                        count += 1
        
        logger.info(f"共发现 {file_count} 个音频文件，成功处理了 {count} 个文件")
    
    elif os.path.isfile(args.input):
        # 处理单个音频文件
        logger.info(f"检测到输入是单个文件: {args.input}")
        result = process_audio_file(args.input, args.output)
        if result:
            logger.info("文件处理成功")
        else:
            logger.error("文件处理失败")
    
    else:
        logger.error(f"输入路径无效: {args.input}")
        print(f"错误: 输入路径无效: {args.input}")

if __name__ == "__main__":
    logger.info("程序入口点，开始执行主程序")
    try:
        main()
        logger.info("程序执行完成")
    except KeyboardInterrupt:
        logger.info("收到键盘中断，程序退出")
    except Exception as e:
        logger.critical(f"程序异常退出: {str(e)}", exc_info=True)
    finally:
        logger.info("程序退出") 