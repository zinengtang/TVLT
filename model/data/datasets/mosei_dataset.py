
import random
import torch
import io
import os
import glob
import json

import pandas as pd
import numpy as np

from model.data.datasets.rawvideo_utils import RawVideoExtractor
from .base_video_dataset import BaseVideoDataset
    
def a2_parse(a):
    if a < 0:
            res = 0
    else: 
            res = 1
    return res


class MOSEIDataset(BaseVideoDataset):
    def __init__(self, *args, split="", **kwargs):
        self.split = split

        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        super().__init__(*args, **kwargs,)        
        
    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.keys)
   
    def _load_metadata(self):
        if self.split=='train':
                          
            self.labels_score = pd.read_csv(self.metadata_dir+'labels/label_file_train.csv')
            self.keys = list(self.labels_score['FileName'])
            self.metadata_dir = self.metadata_dir + 'train/'  
            
            
        elif self.split=='val':
            
            self.labels_score = pd.read_csv(self.metadata_dir+'labels/label_file_valid.csv')
            self.keys = list(self.labels_score['FileName'])
            self.metadata_dir = self.metadata_dir + 'valid/'
            
        elif self.split=='test':
             
            self.labels_score = pd.read_csv(self.metadata_dir+'labels/label_file_test.csv')
            self.keys = list(self.labels_score['FileName'])
            self.metadata_dir = self.metadata_dir + 'test/'           

    def get_suite(self, index):
        result = None
        video_path = self.metadata_dir+'video/'+self.keys[index]+'.mp4'
        audio_path = self.metadata_dir+'audio_wav/'+self.keys[index]+'.wav'
        
        ret = dict()            
        ret.update(self._get_video(index, video_path))
        if self.use_audio:
            ret.update(self._get_audio(index, audio_path))       
        for i in range(self.draw_false_video):
            random_index = random.randint(0, len(self.index_mapper) - 1)
            video_path_f = self.metadata_dir+'video/'+self.keys[random_index]+'.mp4'
            ret.update(self._get_false_video(i, video_path_f))

        score = float(self.labels_score[self.labels_score['FileName']==self.keys[index]]['sentiment_score'])
        label2 = a2_parse(score)
        ret.update({"label2": label2, "score": score})
        
        return ret
