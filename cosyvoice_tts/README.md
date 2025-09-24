# CosyVoice文字转语音系统

基于魔搭平台提供的CosyVoice模型，实现高质量、低延迟的文字转语音功能。

## 功能特点

### 多语言支持
- 中文、英文、日文、韩文
- 多种中文方言（粤语、四川话、上海话、天津话、武汉话等）
- 跨语言和混合语言语音合成

### 低延迟性能
- 双向流式推理支持
- 首包合成延迟低至150ms

### 高准确率
- 相比CosyVoice 1.0，发音错误率降低30%-50%
- 在Seed-TTS评估集的困难测试集上达到最低字符错误率

### 稳定性强
- 确保零样本和跨语言语音合成中可靠的音色一致性
- 跨语言合成相比1.0版本有显著改进

### 自然体验
- 增强的韵律和音质
- 支持更精细的情感控制和口音调整

## 环境配置

### 1. 安装Conda
请参考官方文档安装Miniconda或Anaconda：[https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html)

### 2. 创建Conda环境
```bash
conda create -n cosyvoice python=3.10
conda activate cosyvoice

# 安装pynini（WeTextProcessing的依赖）
conda install -y -c conda-forge pynini==2.1.5

# 安装项目依赖
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host=mirrors.aliyun.com
```

### 3. 安装音频处理依赖
如果遇到sox兼容性问题，请安装：

**Ubuntu:**
```bash
sudo apt-get install sox libsox-dev
```

**CentOS:**
```bash
sudo yum install sox sox-devel
```

**macOS:**
```bash
brew install sox
```

## 模型下载

### 方法一：使用Python代码下载
可以使用我们提供的API自动下载模型：

```python
from cosyvoice_tts import CosyVoiceTTS

# 初始化实例
tts = CosyVoiceTTS()

# 下载所需模型
# 推荐的主要模型
# tts.download_model('CosyVoice2-0.5B')  # 推荐的高性能模型
# tts.download_model('CosyVoice-300M')   # 轻量级模型

# 其他可用模型
# tts.download_model('CosyVoice-300M-SFT')
# tts.download_model('CosyVoice-300M-Instruct')
# tts.download_model('CosyVoice-ttsfrd')  # 文本规范化资源
```

### 方法二：使用Git下载（需要安装git lfs）
```bash
# 确保已安装git lfs
git lfs install

# 创建模型目录
mkdir -p pretrained_models

# 下载模型
git clone https://www.modelscope.cn/iic/CosyVoice2-0.5B.git pretrained_models/CosyVoice2-0.5B
git clone https://www.modelscope.cn/iic/CosyVoice-300M.git pretrained_models/CosyVoice-300M
git clone https://www.modelscope.cn/iic/CosyVoice-ttsfrd.git pretrained_models/CosyVoice-ttsfrd
```

### 安装ttsfrd（可选）
对于更好的文本规范化性能，可以安装ttsfrd包：

```bash
cd pretrained_models/CosyVoice-ttsfrd/
unzip resource.zip -d .
pip install ttsfrd_dependency-0.1-py3-none-any.whl
pip install ttsfrd-0.4.2-cp310-cp310-linux_x86_64.whl
```

**注意**：如果不安装ttsfrd包，系统将默认使用WeTextProcessing。

## 使用示例

### 基本文本转语音
```python
from cosyvoice_tts import CosyVoiceTTS

# 初始化TTS引擎（使用高性能的v2版本）
tts = CosyVoiceTTS(model_path='pretrained_models/CosyVoice2-0.5B', model_type='v2')

# 转换文本为语音并保存
text = "欢迎使用CosyVoice文字转语音系统，这是一个基于深度学习的语音合成引擎。"
tts.text_to_speech(text, output_file='output/basic_tts.wav')

# 如果要复现官网demo的结果
# tts.text_to_speech(text, output_file='output/demo_tts.wav', text_frontend=False)
```

### 零样本声音克隆
```python
# 需要提供参考音频和对应的文本
prompt_audio = 'path/to/reference.wav'  # 参考音频文件
prompt_text = '这是一段用于声音克隆的参考文本。'  # 参考音频对应的文本

# 使用参考音频的声音来合成新文本
new_text = "这是用克隆的声音说出的新内容。"
tts.text_to_speech(new_text, 
                  output_file='output/cloned_voice.wav', 
                  prompt_speaker=prompt_audio, 
                  prompt_text=prompt_text)
```

### 批量文本处理
```python
# 批量处理多条文本
texts = [
    "第一条需要转换的文本。",
    "第二条需要转换的文本。",
    "第三条需要转换的文本。"
]

# 批量生成音频并保存到指定目录
success_files = tts.batch_text_to_speech(texts, 'output/batch_output')
print(f"成功生成的音频文件: {success_files}")
```

### 设置流式推理模式
```python
# 启用流式推理模式（低延迟）
tts.set_streaming_mode(enable=True)

# 使用流式模式生成语音
text = "这是一段使用流式推理模式生成的文本。"
tts.text_to_speech(text, output_file='output/streaming_tts.wav')
```

## 常见问题

1. **模型下载失败**
   - 确保网络连接正常
   - 尝试使用不同的下载方法（Python API或Git）
   - 检查磁盘空间是否充足

2. **依赖安装问题**
   - pynini包建议使用conda安装以获得更好的兼容性
   - 对于macOS用户，可能需要使用brew安装sox

3. **语音合成质量问题**
   - 推荐使用CosyVoice2-0.5B模型以获得最佳性能
   - 尝试调整text_frontend参数
   - 对于特定文本，可能需要调整发音或使用不同的模型

4. **内存不足问题**
   - 如果内存不足，建议使用更轻量级的CosyVoice-300M模型
   - 尝试启用fp16参数以减少内存使用

## 模型版本说明

- **CosyVoice2-0.5B**：最新版本，性能最佳，推荐使用
- **CosyVoice-300M**：轻量级基础模型
- **CosyVoice-300M-SFT**：经过监督微调的模型
- **CosyVoice-300M-Instruct**：指令微调模型，适合对话场景
- **CosyVoice-ttsfrd**：文本规范化资源包

## 注意事项

- 本系统基于魔搭平台提供的CosyVoice模型构建
- 模型仅供研究和学习使用，商业使用请联系模型作者
- 使用过程中如遇到问题，请参考官方文档或提交issue