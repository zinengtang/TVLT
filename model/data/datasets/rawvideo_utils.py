import math
import random
import time

import torch
import torch.nn.functional as F
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import BertTokenizer

from PIL import Image
import torchaudio
from decord import VideoReader, AudioReader
from decord import cpu, gpu
import librosa
import audiosegment
from moviepy.editor import AudioFileClip
import ffmpeg


def time_to_indices(video_reader, time):
    times = video_reader.get_frame_timestamp(range(len(video_reader))).mean(-1)
    indices = np.searchsorted(times, time)
    # Use `np.bitwise_or` so it works both with scalars and numpy arrays.
    return np.where(np.bitwise_or(indices == 0, times[indices] - time <= time - times[indices - 1]), indices,
                    indices - 1)

def pre_spec(x):
    x= normalize(librosa.power_to_db(x) - 20.0)
    return x

def post_spec(x):
    return librosa.db_to_power(denormalize(x) + 20.0)

def normalize(x):
    return np.clip(x / 40.0, -2.0, 0.0) + 1.0

def denormalize(x):
    return (np.clip(x, -1.0, 1.0) - 1.0) * 40.0


def crop_image_only_outside(img, tol=30.0):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img>tol
    mask = mask.all(0).all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[:, row_start:row_end,col_start:col_end]


def center_crop(img, new_width=None, new_height=None):        

    width = img.shape[2]
    height = img.shape[1]

    if new_width is None:
        new_width = min(width, height)

    if new_height is None:
        new_height = min(width, height)

    left = int(np.ceil((width - new_width) / 2))
    right = width - int(np.floor((width - new_width) / 2))

    top = int(np.ceil((height - new_height) / 2))
    bottom = height - int(np.floor((height - new_height) / 2))
    if len(img.shape) == 3:
        center_cropped_img = img[:, top:bottom, left:right]
    else:
        center_cropped_img = img[:, top:bottom, left:right, ...]

    return center_cropped_img


def preprocess_audio(audio, sr):
    audio = audio - audio.mean()        
    audio = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
    audio = torch.from_numpy(pre_spec(audio)).unsqueeze(0)        
    p = 16 - audio.shape[2]%16
    if p > 0:
        audio = F.pad(audio, (0, p, 0, 0), "constant", -1.0)                    
    audio = audio.transpose(1,2)
    return audio    
        
    
class RawVideoExtractor():
    
    def __init__(self, centercrop=True, audio_size=1024, video_size=224, framerate=1, num_frames=8):
        self.centercrop = centercrop
        self.audio_size = audio_size
        self.video_size = video_size
        self.framerate = framerate
        self.max_frames = num_frames
        self.transform_video = self._transform(self.video_size)
        self.sr = 44100
        self.print_error = False
        if not self.print_error:
            import warnings
            warnings.filterwarnings("ignore")
        
    def _transform(self, n_px):
        return Compose([
            Resize([n_px, n_px], interpolation=Image.BICUBIC),])

    def _transform_audio(self, n_px):
        return Normalize(mean=[0.5], std=[0.5])  

    def audio_to_tensor(self, path, timestamp=None):

        try:
            if path.endswith('mp3') or path.endswith('wav') or path.endswith('flac'):
                audio, org_sr = torchaudio.load(path)
                if org_sr != self.sr:
                    audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=self.sr)
                audio = audio.mean(0).numpy()      
                if timestamp is not None:
                    start, end = int(sr * timestamp[0]), int(sr * timestamp[1])
                    audio = audio[start: end]

                audio = preprocess_audio(audio, sr=self.sr)
                audio = audio[:, :self.audio_size]

            elif path.endswith('avi') or path.endswith('mp4'):
                audio = AudioFileClip(path)
                org_sr = audio.fps
                if timestamp is not None:
                    audio = audio.subclip(timestamp[0], timestamp[1]).to_soundarray(fps=org_sr).mean(1)
                else:
                    audio = audio.to_soundarray(fps=org_sr).mean(1)  

                if org_sr != self.sr:
                    audio = torchaudio.functional.resample(torch.tensor(audio), orig_freq=org_sr, new_freq=self.sr).numpy()
                audio = preprocess_audio(audio, sr=self.sr)[:, :self.audio_size]

            else:
                if path.endswith('jpg'):
                    audio = np.array(Image.open(path))/255.0
                    audio = torch.from_numpy(audio*2.0-1.0).unsqueeze(0)
                    audio = audio[:, :, :self.audio_size].transpose(1,2)
                    audio = torch.cat([audio, torch.ones_like(audio[:, :, :32])*-0.794754], -1)
                else:
                    start, end = int(self.sr * timestamp[0]/511.99058), int(self.sr * timestamp[1]/511.99058)
                    index1, index2 = int(start // 65500), int(end // 65500)

                    if index1 == index2:
                        audio = np.array(Image.open(f'{path}_num{str(index1)}.jpg'))/255.0
                        audio = audio[:, int(start%65500): int(end%65500)]
                    else:
                        audio = np.array(Image.open(f'{path}_num{str(index1)}.jpg'))/255.0
                        audio_ = np.array(Image.open(f'{path}_num{str(index2)}.jpg'))/255.0
                        audio = np.concatenate([audio[:, int(start%65500):], audio_[:, :int(end%65500)]], -1)
                    audio = torch.from_numpy(audio*2.0-1.0).unsqueeze(0)
                    p = 16 - audio.shape[2]%16
                    if p > 0:
                        audio = F.pad(audio, (0, p, 0, 0), "constant", -1.0)        
                    audio = audio[:, :, :self.audio_size].transpose(1,2)
        except Exception as e:
            if self.print_error:
                print(e)
            audio = torch.ones([1, 16, 128]) * -1
                
        return audio
        
    def video_to_tensor(self, path, timestamp=None, get_video=True, get_audio=True):

        try:
            video = VideoReader(path)

            framerate = video.get_avg_fps()
            video_len = len(video)/framerate

            if timestamp is not None:
                start, end = time_to_indices(video, timestamp)
                end = min(len(video)-1, end)
                start = min(start, end-1)
                downsamlp_indices = np.linspace(start, end, self.max_frames, endpoint=False).astype(np.int)

            else:                       
                downsamlp_indices = np.linspace(0, len(video), self.max_frames, endpoint=False).astype(np.int)

            video = video.get_batch(downsamlp_indices).asnumpy()
            video = crop_image_only_outside(video)
            min_shape = min(video.shape[1:3])
            video = center_crop(video, min_shape, min_shape)
            video = torch.from_numpy(video).permute(0, 3, 1, 2)
            video = self.transform_video(video)
            video = (video/255.0-0.5)/0.5
        except Exception as e:
            if self.print_error:
                print(e)
            video = torch.ones([self.max_frames, 3, self.video_size, self.video_size]) * -1

        return video
    
    
    def video_audio_to_tensor(self, path, timestamp=None):

        try:
            video = VideoReader(path)
            framerate = video.get_avg_fps()
            video_len = len(video)/video.get_avg_fps()

            if timestamp is not None:
                start, end = time_to_indices(video, timestamp)            
                end = min(len(video)-1, end)
                start = min(start, end-1)
                downsamlp_indices = np.linspace(start, end, self.max_frames, endpoint=False).astype(np.int)
            else:            
                downsamlp_indices = np.linspace(0, len(video), self.max_frames, endpoint=False).astype(np.int)

            video = video.get_batch(downsamlp_indices).asnumpy()
            video = crop_image_only_outside(video)
            min_shape = min(video.shape[1:3])
            video = center_crop(video, min_shape, min_shape)
            video = torch.from_numpy(video).permute(0, 3, 1, 2)
            video = self.transform_video(video)       
            video = (video/255.0-0.5)/0.5

            audio = AudioFileClip(path)   
            sr = audio.fps
            if timestamp is not None:
                audio = audio.subclip(timestamp[0], timestamp[1]).to_soundarray(fps=sr).mean(1)
            else:
                audio = audio.to_soundarray(fps=sr).mean(1)
            audio = preprocess_audio(audio, sr)[:, :self.audio_size]
        except Exception as e:
            print(e)
            audio = torch.zeros([1, 16, 128])
            video = torch.zeros([self.max_frames, 3, self.video_size, self.video_size])
        return video, audio



def _transform(n_px):
    return Compose([
        Resize([n_px, n_px], interpolation=Image.BICUBIC),])
    

def load_audio(path, sr=44100, timestamp=None):
    audio, org_sr = torchaudio.load(path)
    if org_sr != sr:
        audio = torchaudio.functional.resample(audio, orig_freq=org_sr, new_freq=sr)
    audio = audio.mean(0).numpy()      
    if timestamp is not None:
        start, end = int(sr * timestamp[0]), int(sr * timestamp[1])
        audio = audio[start: end]
    audio = preprocess_audio(audio, sr=sr)
    audio = audio[:, :1024]
    return audio.unsqueeze(0).float()

    

def load_video(path, num_frames=8, timestamp=None):
    video = VideoReader(path)
    framerate = video.get_avg_fps()
    video_len = len(video)/framerate

    if timestamp is not None:
        start, end = time_to_indices(video, timestamp)
        end = min(len(video)-1, end)
        start = min(start, end-1)
        downsamlp_indices = np.random.choice(list(range(start, end)), num_frames)

    else:                       
        downsamlp_indices = np.linspace(0, len(video), num_frames, endpoint=False).astype(np.int)

    video = video.get_batch(downsamlp_indices).asnumpy()
    video = crop_image_only_outside(video)
    min_shape = min(video.shape[1:3])
    video = center_crop(video, min_shape, min_shape)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)
    video = _transform(224)(video)
    video = (video/255.0-0.5)/0.5
    return video.unsqueeze(0).float()

    
def load_video_raw(path, num_frames=8, timestamp=None):
    video = VideoReader(path)
    framerate = video.get_avg_fps()
    video_len = len(video)/framerate

    if timestamp is not None:
        start, end = time_to_indices(video, timestamp)
        end = min(len(video)-1, end)
        start = min(start, end-1)
        downsamlp_indices = np.random.choice(list(range(start, end)), num_frames)

    else:                       
        downsamlp_indices = np.linspace(0, len(video), num_frames, endpoint=False).astype(np.int)

    video = video.get_batch(downsamlp_indices).asnumpy()
    video = crop_image_only_outside(video)
    return video
    
    
def load_video_audio(path, num_frames=8, sr=44100, timestamp=None):

    video = VideoReader(path)
    framerate = video.get_avg_fps()
    video_len = len(video)/video.get_avg_fps()

    if timestamp is not None:
        start, end = time_to_indices(video, timestamp)            
        end = min(len(video)-1, end)
        start = min(start, end-1)

        downsamlp_indices = np.random.choice(list(range(start, end)), num_frames)
    else:            
        downsamlp_indices = np.linspace(0, len(video), num_frames, endpoint=False).astype(np.int)

    video = video.get_batch(downsamlp_indices).asnumpy()
    video = crop_image_only_outside(video)
    min_shape = min(video.shape[1:3])
    video = center_crop(video, min_shape, min_shape)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)
    video = _transform(224)(video)       
    video = (video/255.0-0.5)/0.5

    audio = AudioFileClip(path)   
    org_sr = audio.fps
    
    if timestamp is not None:
        audio = audio.subclip(timestamp[0], timestamp[1]).to_soundarray(fps=org_sr).mean(1)
    else:
        audio = audio.to_soundarray(fps=sr).mean(1)
    if org_sr != sr:
        audio = torchaudio.functional.resample(torch.tensor(audio), orig_freq=org_sr, new_freq=sr).numpy()
    audio = preprocess_audio(audio, sr)[:, :1024]
    return video.unsqueeze(0).float(), audio.unsqueeze(0).float()
    

def load_text(path):
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    with open(path) as f:
        text = f.readline()
    encoding = tokenizer(
                text,
                padding="max_length",
                truncation=True,
                max_length=40,
                return_special_tokens_mask=True,
            )
    return text, torch.tensor(encoding['input_ids']).unsqueeze(0), torch.tensor(encoding['attention_mask']).unsqueeze(0), 

def load_image(path, image_size=224):
    
    image = np.array(Image.open(path).convert("RGB").resize((image_size, image_size), Image.ANTIALIAS))
    image = image/255.0*2.0-1.0
    image = torch.tensor(image[np.newaxis, ...]).permute(0, 3, 1, 2)
    return image    
