import json
import os

def load_config(config_path):
    """설정 파일을 로드합니다."""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def save_model(model, path):
    """모델의 가중치를 저장합니다."""
    model.save_weights(path)

def load_model(model, path):
    """모델의 가중치를 로드합니다."""
    model.load_weights(path)
