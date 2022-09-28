import random
import torch
import io
import os
import glob
from tqdm import tqdm
import json

import pandas as pd
import numpy as np
from model.data.datasets.rawvideo_utils import RawVideoExtractor
from .base_video_dataset import BaseVideoDataset

class Howto100mDataset(BaseVideoDataset):
    def __init__(self, *args, split="", **kwargs):
        self.split = split

        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        self.keys = None
        super().__init__(*args, **kwargs,)  
        
        
    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.keys)
   
    def _load_metadata(self):
        print("loading metadata for howto100m")
        
        if not os.path.exists(os.path.join(self.metadata_dir, 'ht_caption_train.json')):
            captions = json.load(open(os.path.join(self.metadata_dir, 'caption.json')))           
            new_caption = {}

            video_keys = []
            exist_videos = list(os.listdir(os.path.join(self.metadata_dir, 'videos_ht/')))
            for video in tqdm(exist_videos):
                video_keys += [video.split('.')[0]]
            all_keys = video_keys
            for key in tqdm(all_keys):
                new_caption[key] = captions[key]

            all_samples = list(new_caption.items())
            random.shuffle(all_samples)
            caption_train = dict(all_samples[:-1000])
            caption_val = dict(all_samples[-1000:])

            json.dump(caption_train, open(os.path.join(self.metadata_dir, 'ht_caption_train.json')), 'w')
            json.dump(caption_val, open(os.path.join(self.metadata_dir, 'ht_caption_val.json')), 'w')

        if self.split=='train':
            self.metadata = json.load(open(os.path.join(self.metadata_dir, 'ht_caption_train.json')))
        else:
            self.metadata = json.load(open(os.path.join(self.metadata_dir, 'ht_caption_val.json')))

    def find_overlap(self, s1, s2):
        for i in range(len(s1)):
            test1, test2 = s1[i:], s2[:len(s1) - i]
            if test1 == test2:
                return s1[:i], s2            
        return s1, s2
    
    def preprocess_text(self, asr_samples):
        text = []
        for i in range(len(asr_samples)-1):
            s1, s2 = self.find_overlap(str(asr_samples[i]), str(asr_samples[i+1]))
            if s1.strip():
                text += [s1.strip()]
        if s2.strip():
            text += [s2.strip()]
        return ' . '.join(text)

    def get_suite(self, index):
        result = None
        sample = self.metadata[self.keys[index]]
        sample_idx = random.choice(range(len(sample['start'])))
        video_path = os.path.join(self.metadata_dir, 'videos_ht/', self.keys[index]+'.mp4')
        timestamp = [sample['start'][max(sample_idx-1, 0)], sample['end'][min(sample_idx+1, len(sample['start'])-1)]] 
        

        ret = dict()     
        ret.update(self._get_video_audio(index, video_path, timestamp))

        if self.use_text:
            text = self.preprocess_text([sample['text'][max(sample_idx-1, 0)], sample['text'][sample_idx], sample['text'][min(sample_idx+1, len(sample['text'])-1)]])
            ret.update(self._get_text(index, text))                                
            for i in range(self.draw_false_text):
                random_index = random.randint(0, len(self.index_mapper) - 1)
                sample_f = self.metadata[self.keys[random_index]]
                sample_idx_f = random.choice(range(len(sample_f['start'])))
                text_f = sample_f['text'][sample_idx_f]
                ret.update(self._get_false_text(i, text_f))

        for i in range(self.draw_false_video):
            random_index = random.randint(0, len(self.keys) - 1)
            sample_f = self.metadata[self.keys[random_index]]
            sample_idx_f = random.choice(range(len(sample_f['start'])))
            timestamp_f = [sample_f['start'][max(sample_idx_f-1, 0)], sample_f['end'][min(sample_idx_f+1, len(sample_f['start'])-1)]] 
            video_path_f = os.path.join(self.metadata_dir, 'videos_ht/', self.keys[random_index]+'.mp4')
            ret.update(self._get_false_video(i, video_path_f, timestamp_f))

        return ret
