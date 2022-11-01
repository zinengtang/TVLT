
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


class MOSEIEMODataset(BaseVideoDataset):
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
                
            self.labels = pd.read_csv(self.metadata_dir+'labels_emotion/label_file_train.csv')
            self.keys = list(self.labels['FileName'])
            self.metadata_dir = self.metadata_dir + 'train/'       
            
        elif self.split=='val':
            
            self.labels = pd.read_csv(self.metadata_dir+'labels_emotion/label_file_valid.csv')
            self.keys = list(self.labels['FileName'])
            self.metadata_dir = self.metadata_dir + 'valid/'
            
        elif self.split=='test':
             
            self.labels = pd.read_csv(self.metadata_dir+'labels_emotion/label_file_test.csv')
            self.keys = list(self.labels['FileName'])
            self.metadata_dir = self.metadata_dir + 'test/'           
       

    def get_suite(self, index):
        result = None
        video_path = self.metadata_dir+'video/'+self.keys[index]+'.mp4'
        audio_path = self.metadata_dir+'audio_wav/'+self.keys[index]+'.wav'
        
        ret = dict()            
        ret.update(self._get_video(index, video_path, rand_sample=True))
        if self.use_audio:
            ret.update(self._get_audio(index, audio_path)) 

        if self.use_text:
            text = sample['text']
            ret.update(self._get_text(index, text))                                
            for i in range(self.draw_false_text):
                random_index = random.randint(0, len(self.index_mapper) - 1)
                sample_f = self.metadata[self.keys[random_index]]
                text_f = sample_f['text']
                ret.update(self._get_false_text(i, text_f))

        for i in range(self.draw_false_video):
            random_index = random.randint(0, len(self.index_mapper) - 1)
            video_path_f = self.metadata_dir+'video/'+self.keys[random_index]+'.mp4'
            ret.update(self._get_false_video(i, video_path_f, rand_sample=True))

        emolist = np.array(self.labels[self.labels['FileName']==self.keys[index]].iloc[0, 3:] > 0.0)
        ret.update({"emolist": emolist})

        return ret
