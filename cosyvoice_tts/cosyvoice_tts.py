import sys
import os
import torchaudio
from cosyvoice.cli.cosyvoice import CosyVoice, CosyVoice2
from cosyvoice.utils.file_utils import load_wav
from tqdm import tqdm

class CosyVoiceTTS:
    """
    CosyVoice文字转语音功能封装类
    
    该类封装了魔搭平台提供的CosyVoice模型的文字转语音功能，支持多语言、零样本声音克隆、流式推理等特性。
    支持的语言包括：中文、英文、日文、韩文以及多种中文方言。
    """
    
    def __init__(self, model_path='pretrained_models/CosyVoice2-0.5B', 
                 model_type='v2', load_jit=False, load_trt=False, fp16=False):
        """
        初始化CosyVoiceTTS类
        
        Args:
            model_path (str): 预训练模型路径，默认为'pretrained_models/CosyVoice2-0.5B'
            model_type (str): 模型类型，支持'v1'或'v2'，默认为'v2'（推荐使用v2版本）
            load_jit (bool): 是否加载JIT优化模型，默认为False
            load_trt (bool): 是否加载TensorRT优化模型，默认为False
            fp16 (bool): 是否使用半精度浮点数，默认为False
        
        Returns:
            None
        """
        # 添加Matcha-TTS到系统路径
        third_party_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'third_party', 'Matcha-TTS')
        if third_party_path not in sys.path:
            sys.path.append(third_party_path)
        
        # 初始化模型
        self.model_path = model_path
        self.model_type = model_type
        
        try:
            if model_type == 'v2':
                self.cosyvoice = CosyVoice2(model_path, load_jit=load_jit, load_trt=load_trt, fp16=fp16)
            else:
                self.cosyvoice = CosyVoice(model_path, load_jit=load_jit, load_trt=load_trt, fp16=fp16)
            print(f"成功加载{model_type}模型: {model_path}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            print("请确保已正确下载模型并安装依赖库")
            self.cosyvoice = None
    
    def text_to_speech(self, text, output_file=None, text_frontend=True, 
                      prompt_speaker=None, prompt_text=None):
        """
        将文本转换为语音
        
        Args:
            text (str): 要转换的文本
            output_file (str, optional): 输出音频文件路径，默认为None（不保存文件）
            text_frontend (bool): 是否使用文本前端处理，默认为True
            prompt_speaker (str, optional): 提示说话人的音频文件路径，用于零样本声音克隆
            prompt_text (str, optional): 提示说话人的文本内容，与prompt_speaker配合使用
        
        Returns:
            tuple: (音频数据, 采样率)，如果处理失败则返回(None, None)
        
        Notes:
            - 如果要复现官网demo的结果，请设置text_frontend=False
            - 当提供prompt_speaker和prompt_text时，将使用零样本声音克隆功能
        """
        if not self.cosyvoice or not text:
            return None, None
        
        try:
            # 判断是否使用零样本声音克隆
            if prompt_speaker and prompt_text:
                # 零样本声音克隆
                audio, sample_rate = self.cosyvoice.infer_from_prompt(
                    prompt_speaker=prompt_speaker,
                    prompt_text=prompt_text,
                    text=text,
                    text_frontend=text_frontend
                )
            else:
                # 普通文本转语音
                audio, sample_rate = self.cosyvoice.infer(
                    text=text,
                    text_frontend=text_frontend
                )
            
            # 保存音频文件
            if output_file:
                os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
                torchaudio.save(output_file, audio, sample_rate)
                print(f"音频已保存至: {output_file}")
            
            return audio, sample_rate
        except Exception as e:
            print(f"文本转语音失败: {e}")
            return None, None
    
    def batch_text_to_speech(self, texts, output_dir, text_frontend=True):
        """
        批量将文本转换为语音
        
        Args:
            texts (list): 文本列表
            output_dir (str): 输出音频文件目录
            text_frontend (bool): 是否使用文本前端处理，默认为True
        
        Returns:
            list: 成功处理的音频文件路径列表
        """
        if not self.cosyvoice or not texts:
            return []
        
        os.makedirs(output_dir, exist_ok=True)
        success_files = []
        
        for i, text in enumerate(texts):
            output_file = os.path.join(output_dir, f"tts_output_{i}.wav")
            audio, sample_rate = self.text_to_speech(text, output_file, text_frontend)
            if audio is not None:
                success_files.append(output_file)
        
        return success_files
    
    def set_streaming_mode(self, enable=True):
        """
        设置流式推理模式
        
        Args:
            enable (bool): 是否启用流式推理，默认为True
        
        Returns:
            bool: 设置是否成功
        
        Notes:
            - 流式推理模式可以实现低延迟的语音合成
            - CosyVoice 2.0版本支持双向流式推理
        """
        if not self.cosyvoice:
            return False
        
        try:
            # 这里假设CosyVoice模型提供了设置流式模式的接口
            # 具体实现可能需要根据实际模型API进行调整
            if hasattr(self.cosyvoice, 'set_streaming_mode'):
                self.cosyvoice.set_streaming_mode(enable)
                return True
            else:
                print("当前模型版本不支持流式推理模式设置")
                return False
        except Exception as e:
            print(f"设置流式推理模式失败: {e}")
            return False
    
    def download_model(self, model_name='CosyVoice2-0.5B', save_dir='pretrained_models', show_progress=True):
        """
        下载预训练模型（带进度条）
        
        Args:
            model_name (str): 模型名称，支持'CosyVoice2-0.5B', 'CosyVoice-300M', 'CosyVoice-300M-SFT', 'CosyVoice-300M-Instruct', 'CosyVoice-ttsfrd'
            save_dir (str): 保存目录，默认为'pretrained_models'
            show_progress (bool): 是否显示下载进度条，默认为True
        
        Returns:
            bool: 下载是否成功
        
        Notes:
            - 此函数依赖modelscope库和tqdm库，请确保已安装
            - 推荐优先下载CosyVoice2-0.5B以获得更好的性能
        """
        try:
            from modelscope import snapshot_download
            
            save_path = os.path.join(save_dir, model_name)
            os.makedirs(save_dir, exist_ok=True)
            
            print(f"开始下载模型: {model_name}")
            
            # 使用tqdm显示进度条
            if show_progress:
                # 创建进度条回调函数
                def progress_callback(current, total):
                    if total > 0:
                        progress = current / total * 100
                        print(f"\r下载进度: {progress:.2f}% ({current}/{total} MB)", end="")
                
                # 设置进度条
                snapshot_download(f'iic/{model_name}', local_dir=save_path, progressbar=show_progress)
                print()  # 换行，确保进度条完成后输出到下一行
            else:
                snapshot_download(f'iic/{model_name}', local_dir=save_path, progressbar=False)
            
            print(f"模型下载成功，保存至: {save_path}")
            
            # 如果是ttsfrd资源，提供安装提示
            if model_name == 'CosyVoice-ttsfrd':
                print("\n提示：您可以选择解压ttsfrd资源并安装ttsfrd包以获得更好的文本规范化性能：")
                print(f"cd {save_path}")
                print("unzip resource.zip -d .")
                print("pip install ttsfrd_dependency-0.1-py3-none-any.whl")
                print("pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl")
                print("\n注意：如果不安装ttsfrd包，系统将默认使用WeTextProcessing")
            
            return True
        except ImportError as e:
            if 'tqdm' in str(e):
                print("请先安装tqdm库: pip install tqdm")
            else:
                print("请先安装modelscope库: pip install modelscope")
            return False
        except Exception as e:
            print(f"模型下载失败: {e}")
            return False

if __name__ == '__main__':
    """
    示例用法
    """
    # 初始化TTS引擎
    tts = CosyVoiceTTS(model_path='pretrained_models/CosyVoice2-0.5B', model_type='v2')
    
    # 如果模型未下载，可以使用以下代码下载
    # tts.download_model('CosyVoice2-0.5B')
    # tts.download_model('CosyVoice-ttsfrd')
    
    # 基本文本转语音
    text = "欢迎使用CosyVoice文字转语音系统，这是一个基于深度学习的语音合成引擎。"
    tts.text_to_speech(text, output_file='output/basic_tts.wav')
    
    # 零样本声音克隆示例（需要提供参考音频和文本）
    # prompt_audio = 'path/to/reference.wav'
    # prompt_text = '这是一段参考文本，用于声音克隆。'
    # tts.text_to_speech(text, output_file='output/cloned_voice.wav', 
    #                    prompt_speaker=prompt_audio, prompt_text=prompt_text)
    
    # 批量处理示例
    # texts = ["第一条文本", "第二条文本", "第三条文本"]
    # tts.batch_text_to_speech(texts, 'output/batch_output')