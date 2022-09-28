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
from .base_image_dataset import BaseImageDataset


def load_jsonl(path):
    jsonl_content = open(path).read()
    result = [json.loads(jline) for jline in jsonl_content.splitlines()]
    return result


ANS_CONVERT = {
        "a man": "man",
        "the man": "man",
        "a woman": "woman",
        "the woman": "woman",
        'one': '1',
        'two': '2',
        'three': '3',
        'four': '4',
        'five': '5',
        'six': '6',
        'seven': '7',
        'eight': '8',
        'nine': '9',
        'ten': '10',
        'grey': 'gray',
    }

def convert_ans(ans):
        if len(ans) == 0:
            return ""
        ans = ans.lower()
        if ans[-1] == '.':
            ans = ans[:-1].strip()
        if ans.startswith("a "):
            ans = ans[2:].strip()
        if ans.startswith("an "):
            ans = ans[3:].strip()
        if ans.startswith("the "):
            ans = ans[4:].strip()
        if ans in ANS_CONVERT:
            ans = ANS_CONVERT[ans]
        return ans
    
def img_id2_path(img_id):
    if img_id.startswith("COCO_train2014"):
        return f"train2014/{img_id}.jpg"
    elif img_id.startswith("COCO_val2014"):
        return f"val2014/{img_id}.jpg"
    elif img_id.startswith("COCO_test2015"):
        return f"test2015/{img_id}.jpg"
    
    
class VQADataset(BaseImageDataset):
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
        return len(self.metadata)
   
    def _load_metadata(self):
        
        self.metadata_dir_audio = os.path.join(self.metadata_dir, 'audios/')
        
        if self.split=='train':
            self.metadata = load_jsonl(os.path.join(self.metadata_dir, 'train.jsonl'))
            
        elif self.split=='val':
            self.metadata = load_jsonl(os.path.join(self.metadata_dir, 'dev.jsonl'))
        else:
            self.metadata = load_jsonl(os.path.join(self.metadata_dir, 'test.jsonl'))
        self.label2ans = json.load(open(os.path.join(self.metadata_dir, 'trainval_ans2label.json')))
        self.keys = None
        
    def get_suite(self, index):
        result = None
        sample = self.metadata[index]
        
        image_path = self.metadata_dir+img_id2_path(sample['img_id'])
        
        audio_path = self.metadata_dir_audio+str(sample['question_id'])+'.mp3'
        
        ret = dict()
        ret.update(self._get_image(index, image_path))
        if self.use_audio:
            ret.update(self._get_audio(index, audio_path))                                   
                           
        labels = list(map(convert_ans, list(sample['label'].keys())))                    
        labels = list(map(self.label2ans.get, labels))
        scores = list(sample['label'].values())
        ret.update({"vqa_labels": labels, 'vqa_scores': scores})
        return ret
