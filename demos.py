import numpy as np
import matplotlib.pyplot as plt

import torch
from huggingface_sb3 import load_from_hub
import model.modules.tvlt as tvlt


def MAE_config():
    return {'exp_name': 'mae_vam', 'loss_names': {'vam': 1, 'vtm': 0, 'mae_audio': 1, 'mae_video': 1, 'vqa': 0, 'mlm': 0, 'mosei': 0, 'moseiemo': 0}, 'max_text_len': 40, 'tokenizer': 'bert-base-uncased', 'vocab_size': 30522, 'whole_word_masking': False, 'mlm_prob': 0.15, 'use_text': False, 'video_size': 224, 'video_only': False, 'max_frames': 64, 'num_frames': 1, 'use_video': True, 'audio_size': 1024, 'frequency_size': 128, 'max_audio_patches': 1020, 'mam_prob': 0.75, 'use_audio': True, 'audio_only': False, 'frame_masking': True, 'model_type': 'mae_vit_base_patch16_dec512d8b', 'patch_size': 16, 'audio_patch_size': [16, 16], 'hidden_size': 768, 'decoder_hidden_size': 512, 'num_heads': 12, 'num_layers': 12, 'mlp_ratio': 4, 'use_mae': True, 'drop_rate': 0.1}


def MOSEI_sentiment_config():
    return {"exp_name": "cls_mosei", "loss_names": {"vam": 0, "vtm": 0, "mae_audio": 0, "mae_video": 0, "vqa": 0, "mlm": 0, "mosei": 1, "moseiemo": 0}, "max_text_len": 40, "draw_false_text": 0, "tokenizer": "bert-base-uncased", "vocab_size": 30522, "whole_word_masking": False, "mlm_prob": 0.15, "use_text": False, "video_size": 224, "max_frames": 64, "num_frames": 8, "use_video": True, "audio_size": 1024, "frequency_size": 128, "max_audio_patches": 1020, "mam_prob": 0.75, "draw_false_audio": 0, "use_audio": True, "frame_masking": False, "model_type": "mae_vit_base_patch16_dec512d8b", "patch_size": 16, "audio_patch_size": [16, 16], "hidden_size": 768, "decoder_hidden_size": 512, "num_heads": 12, "num_layers": 12, "mlp_ratio": 4, "use_mae": False, "drop_rate": 0.1}


def MOSEI_emotion_config():
    return {"exp_name": "cls_moseiemo", "loss_names": {"vam": 0, "vtm": 0, "mae_audio": 0, "mae_video": 0, "vqa": 0, "mlm": 0, "mosei": 0, "moseiemo": 1}, "max_text_len": 40, "draw_false_text": 0, "tokenizer": "bert-base-uncased", "vocab_size": 30522, "whole_word_masking": False, "mlm_prob": 0.15, "use_text": False, "video_size": 224, "max_frames": 64, "num_frames": 8, "use_video": True, "audio_size": 1024, "frequency_size": 128, "max_audio_patches": 1020, "mam_prob": 0.75, "draw_false_audio": 0, "use_audio": True, "frame_masking": False, "model_type": "mae_vit_base_patch16_dec512d8b", "patch_size": 16, "audio_patch_size": [16, 16], "hidden_size": 768, "decoder_hidden_size": 512, "num_heads": 12, "num_layers": 12, "mlp_ratio": 4, "use_mae": False}


def MOSEI_emotion_config_text():
    return {"exp_name": "cls_moseiemo", "loss_names": {"vam": 0, "vtm": 0, "mae_audio": 0, "mae_video": 0, "vqa": 0, "mlm": 0, "mosei": 0, "moseiemo": 1}, "max_text_len": 40, "draw_false_text": 0, "tokenizer": "bert-base-uncased", "vocab_size": 30522, "whole_word_masking": False, "mlm_prob": 0.15, "use_text": True, "video_size": 224, "max_frames": 64, "num_frames": 8, "use_video": True, "audio_size": 1024, "frequency_size": 128, "max_audio_patches": 1020, "mam_prob": 0.75, "draw_false_audio": 0, "use_audio": False, "frame_masking": False, "model_type": "mae_vit_base_patch16_dec512d8b", "patch_size": 16, "audio_patch_size": [16, 16], "hidden_size": 768, "decoder_hidden_size": 512, "num_heads": 12, "num_layers": 12, "mlp_ratio": 4, "use_mae": False, "drop_rate": 0.1}


def MAE_model(model_path=''):
    config = MAE_config()
    model = getattr(tvlt, 'mae_vit_base_patch16_dec512d8b')(
        config=config).float().eval()
    ckpt_path = load_from_hub(repo_id="TVLT/models", filename="TVLT.ckpt")
    model.load_state_dict(torch.load(ckpt_path))
    return model


def MOSEI_sentiment_model(model_path=''):
    config = MOSEI_sentiment_config()
    model = getattr(tvlt, 'mae_vit_base_patch16_dec512d8b')(
        config=config).float().eval()
    ckpt_path = load_from_hub(repo_id="TVLT/models",
                              filename="TVLT-MOSEI-SA.ckpt")
    model.load_state_dict(torch.load(ckpt_path))
    return model


def MOSEI_emotion_model(model_path=''):
    config = MOSEI_emotion_config()
    model = getattr(tvlt, 'mae_vit_base_patch16_dec512d8b')(
        config=config).float().eval()
    ckpt_path = load_from_hub(repo_id="TVLT/models",
                              filename="TVLT-MOSEI-EA.ckpt")
    model.load_state_dict(torch.load(ckpt_path))
    return model


def MOSEI_emotion_model_text(model_path=''):
    config = MOSEI_emotion_config_text()
    model = getattr(tvlt, 'mae_vit_base_patch16_dec512d8b')(
        config=config).float().eval()
    ckpt_path = load_from_hub(repo_id="TVLT/models",
                              filename="TVLT-MOSEI-EA-text.ckpt")
    model.load_state_dict(torch.load(ckpt_path))
    return model


def visualize_video(pred_v):
    video_span = 196
    pred_v_ = pred_v
    b, t, h = pred_v_.shape
    p = np.sqrt(t)
    video = np.transpose(pred_v_.reshape(b, 14, 14, 16, 16, 3), [
                         0, 1, 3, 2, 4, 5]).reshape([b, 14*16, 14*16, 3])
    video = np.clip((video + 1)/2.0, 0.0, 1.0)

    plt.axis('off')
    plt.imshow(video[0].astype(float))
    plt.show()


def visualize_audio(pred_a, audio_len=176):
    def denormalize(x):
        return (np.clip(x, -1.0, 1.0) + 1.0)/2.0
    b, t, h = pred_a.shape
    p = t//8

    audio = np.transpose(pred_a.reshape(b, p, 8, 16, 16), [
                         0, 1, 3, 2, 4]).reshape([b, p*16, 8*16])
    audio = denormalize(audio)

    plt.axis('off')
    plt.imshow(audio[0].astype(float).transpose(1, 0)[:, :audio_len])
    plt.show()
