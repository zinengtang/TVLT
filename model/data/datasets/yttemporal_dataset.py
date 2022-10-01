import random
import torch
import io
import os
from tqdm import tqdm
import json

import pandas as pd
import numpy as np
from model.data.datasets.rawvideo_utils import RawVideoExtractor
from .base_video_dataset import BaseVideoDataset


class YTTemporalDataset(BaseVideoDataset):
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
        print("loading metadata for yttemporal")
        
        if not os.path.exists(os.path.join(self.metadata_dir, 'yt_caption_train.json')):
            train = list(os.listdir(os.path.join(self.metadata_dir, 'videos_yt')))[:-1000]
            val = list(os.listdir(os.path.join(self.metadata_dir, 'videos_yt')))[-1000:]

            json.dump(train, open(os.path.join(self.metadata_dir, 'yt_caption_train.json'), 'w'))
            json.dump(val, open(os.path.join(self.metadata_dir, 'yt_caption_val.json'), 'w'))


        if self.split=='train':
            self.keys = json.load(open(os.path.join(self.metadata_dir, 'yt_caption_train.json')))
        else:
            self.keys = json.load(open(os.path.join(self.metadata_dir, 'yt_caption_val.json'))) 

    
    def get_suite(self, index):
        
        video_path = os.path.join(self.metadata_dir, 'videos_yt', self.keys[index])

        ret = dict()
        ret.update(self._get_video_audio(index, video_path, rand_sample=True))

        for i in range(self.draw_false_video):
            random_index = random.randint(0, len(self.keys) - 1)
            video_path_f = os.path.join(self.metadata_dir, 'videos_yt', self.keys[random_index])
            ret.update(self._get_false_video(i, video_path_f, rand_sample=True))

        return ret
