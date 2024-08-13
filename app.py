from flask import Flask, request, jsonify, render_template, send_from_directory
import os
import random
from pydub import AudioSegment
import torch
import librosa
import json
import glob
from transformers import AutoModelForAudioClassification, AutoFeatureExtractor


app = Flask(__name__)

# 初始化模型和特徵提取器
model_name = "ntu-spml/distilhubert"
feature_extractor = AutoFeatureExtractor.from_pretrained(model_name)
model = AutoModelForAudioClassification.from_pretrained(model_name)
audiopathrn="ifwrongthenshowmeBY陳冠廷"#宜家ge音頻路徑
audiotypenow="ifwrongthenshowmeBY陳冠廷"#宜家ge音頻類型
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

# 修改get_audio_path函数，增加返回对应文字
def get_audio_path():
    """按顺序随机选择 T1 或 F1 文件夹中的 WAV 文件"""
    global current_index
    global audiopathrn
    global audiotypenow
    
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
    
    # 确定音频类型
    if selected_folder == "T1":
        audiotypenow = "r"
    else:
        audiotypenow = "f"
    
    audiopathrn = "static/" + str(audio_path)  # 保存全局变量
    
    # 提取文件名中的编号
    audio_index = str(current_index)
    print(f"Current audio index: {audio_index}")  # 打印当前的音频编号
    
    # 获取对应的文本
    associated_text = t1_text_dict.get(audio_index, "No corresponding text found")
    print(f"Associated Text Found: {associated_text}")  # 打印找到的对应文本
    
    # 增加索引，以便下次选择下一个文件
    current_index += 1
    
    return audio_path, associated_text



@app.route('/game')
def game():
    """处理 /game 请求并渲染模板"""
    audio_path, associated_text = get_audio_path()  # 调用获取路径的功能
    print(f"Serving audio file: {audio_path} with text: {associated_text}")  # 输出调试信息
    return render_template('game2.html', audio_path=audio_path, text=associated_text)


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

@app.route('/predict', methods=['POST'])
def predict():
    aiprediction = predict_full_audio(audiopathrn)
    audiotype=audiotypenow
    return jsonify({'aiprediction': aiprediction, 'audiotype': audiotype})

@app.route('/takeans', methods=['POST'])
def takeans():
    audiotype=audiotypenow
    return jsonify({'audiotype': audiotype}) 

def load_t1_text(file_path):
    text_dict = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line and ": " in line:  # 确保行不为空并且包含 ': '
                try:
                    key, value = line.split(": ", 1)
                    key = key.strip('-')  # 移除前面的'-'
                    text_dict[key] = value
                except ValueError:
                    print(f"Skipping line due to unexpected format: {line}")
            else:
                print(f"Skipping line due to missing ': ': {line}")
    return text_dict


# 将T1.txt文件加载为字典
t1_text_dict = load_t1_text('static\list\T1.txt')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')