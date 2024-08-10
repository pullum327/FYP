from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import random
from pydub import AudioSegment
import torch
import librosa
import glob
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor


app = Flask(__name__)

# 初始化模型和特徵提取器
model_name = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)

def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if 'model_state_dict' in checkpoint and model is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    return None

checkpoint_path = 'model_finetunelaogao_5s_3.0.bin'
load_checkpoint(model, checkpoint_path)

@app.route('/')
def index():
    return render_template('index.html')

current_index = 0

def get_audio_path():
    """按顺序随机选择 T1 或 F1 文件夹中的 WAV 文件"""
    global current_index
    
    # 定义文件夹和文件名的模板
    folder_options = ["T1", "F1"]
    file_template = "{folder}_{index}.wav"
    
    # 随机选择一个文件夹
    selected_folder = random.choice(folder_options)
    
    base_path = "audio/Target2"
    # 构建完整的文件路径
    audio_path = os.path.join(base_path, selected_folder, file_template.format(folder=selected_folder, index=current_index))
    
    audio_path = audio_path.replace("\\", "/")
    print(f"Audio Path: {audio_path}")  # 添加调试信息
    
    # 增加索引，以便下次选择下一个文件
    current_index += 1
    
    return audio_path

@app.route('/game')
def game():
    """处理 /game 请求并渲染模板"""
    audio_path = get_audio_path()  # 调用获取路径的功能
    return render_template('game2.html', audio_path=audio_path)


def predict_full_audio(audio_path):
    waveform, sample_rate = librosa.load(audio_path, sr=16000)  
    total_samples = waveform.shape[0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    inputs = feature_extractor(waveform, sampling_rate=sample_rate, return_tensors="pt", padding="max_length", max_length=total_samples, truncation=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}  
    with torch.no_grad():
        outputs = model(**inputs)
    pred_label = torch.argmax(outputs.logits, dim=-1)
    predicted_description = "f" if pred_label.item() == 0 else "r"
    return predicted_description


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')