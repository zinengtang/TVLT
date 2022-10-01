import random
import torch
import io
import os
import glob
from tqdm import tqdm
import json

import numpy as np
from model.data.datasets.rawvideo_utils import RawVideoExtractor

class BaseVideoDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_dir = '.',
        audio_size=1024,
        video_size=224,
        max_text_len=40,
        num_frames=8,
        draw_false_audio=0,
        draw_false_video=1,
        draw_false_text=0,
        audio_only=False,
        video_only=False,
        use_audio=True,
        use_text=False,
    ):
        """
        data_dir : where dataset file *.arrow lives; existence should be guaranteed via DataModule.prepare_data
        transform_keys : keys for generating augmented views of images
        text_column_name : pyarrow table column name that has list of strings as elements
        """
        super().__init__()
              
        self.metadata_dir = data_dir
        
        self.video_loader = RawVideoExtractor(video_size=video_size, audio_size=audio_size, num_frames=num_frames)

        self.video_only = video_only
        self.video_size = video_size
        self.audio_only = audio_only
        self.audio_size = audio_size
        
        self.use_audio = use_audio
        self.use_text = use_text
        
        self.draw_false_text = draw_false_text 
        self.draw_false_video = draw_false_video
        self.draw_false_audio = draw_false_audio
        
        self.max_text_len = max_text_len
        self.num_frames = num_frames
        self.subsample = 1
        
        self._load_metadata()
        if self.keys is None:
            self.keys = list(self.metadata.keys())
               
        self.index_mapper = list(range(len(self.keys)))
        
    @property
    def corpus(self):
        return [text for texts in self.all_texts for text in texts]

    def __len__(self):
        return len(self.keys)

    def _get_audio_span(self, spans):
        return {'audio_spans': spans}
    
    def _get_video(self, index, video_path, timestamp=None, rand_sample=False):
        video_data = self.video_loader.video_to_tensor(video_path, timestamp, rand_sample=rand_sample)
        return {"video_data": video_data,
                "v_index": index,}   

    def _get_audio(self, index, video_path, timestamp=None):
        audio_data = self.video_loader.audio_to_tensor(video_path, timestamp)
        return {"audio_data": audio_data,
                "raw_index": index,
                "a_index": index,}   
    
    def _get_video_audio(self, index, path, timestamp=None, rand_sample=False):
        video_data, audio_data = self.video_loader.video_audio_to_tensor(path, timestamp, rand_sample=rand_sample)
        return {"video_data": video_data,
                "v_index": index,
                "audio_data": audio_data,
                "raw_index": index,
                "a_index": index,}
    
    def _get_false_video(self, rep, video_path, timestamp=None, rand_sample=False):   
        video_data = self.video_loader.video_to_tensor(video_path, timestamp, rand_sample=rand_sample)
        return {f"false_video_{rep}": video_data}
    
    def _get_false_audio(self, rep, video_path, timestamp=None):        
        audio_data = self.video_loader.audio_to_tensor(video_path, timestamp)
        return {f"false_audio_{rep}": audio_data}
    
    def _encode_text(self, text):
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_text_len,
            return_special_tokens_mask=True,
        )
        return text, encoding
    
    def _get_text(self, index, text):
        text, encoding = self._encode_text(text)
        return {"text": (text, encoding),
                "t_index": index,}    

    def _get_false_text(self, rep, text):
        text, encoding = self._encode_text(text)
        return {f"false_text_{rep}": (text, encoding)}    
    
    def __getitem__(self, index):
        return self.get_suite(index)
    
    def collate(self, batch, mlm_collator):
        batch_size = len(batch)
        keys = set([key for b in batch for key in b.keys()])
        dict_batch = {k: [dic[k] if k in dic else None for dic in batch] for k in keys}

        video_keys = [k for k in list(dict_batch.keys()) if "video" in k]
        video_sizes = list()
        for video_key in video_keys:
            video = dict_batch[video_key]
            video_sizes += [video[0].shape]
        for size in video_sizes:
            assert (
                len(size) == 4
            ), f"Collate error, an video should be in shape of (T, 3, H, W), instead of given {size}"
        if len(video_keys) != 0:
            max_video_length = self.num_frames
            max_height = max([i[2] for i in video_sizes])
            max_width = max([i[3] for i in video_sizes])
        for video_key in video_keys:
            video = dict_batch[video_key]
            new_videos = torch.ones(batch_size, max_video_length, 3, max_height, max_width)*-1.0
            for bi in range(batch_size):
                orig_batch = video[bi]
                if orig_batch is None:
                    new_videos[bi] = None
                else:
                    orig = video[bi]
                    new_videos[bi, : orig.shape[0], :, : orig.shape[2], : orig.shape[3]] = orig
            dict_batch[video_key] = new_videos

            
        audio_keys = [k for k in list(dict_batch.keys()) if "audio" in k]
        audio_sizes = list()
        for audio_key in audio_keys:
            audio = dict_batch[audio_key]
            for audio_i in audio:
                audio_sizes += [audio_i.shape]
        for size in audio_sizes:
            assert (
                len(size) == 3
            ), f"Collate error, an audio should be in shape of (1, H, W), instead of given {size}"
        if len(audio_keys) != 0:
            max_height = max([i[1] for i in audio_sizes])
            max_width = max([i[2] for i in audio_sizes])

        for audio_key in audio_keys:
            audio = dict_batch[audio_key]
            new_audios = torch.ones(batch_size, 1, max_height, max_width)*-1.0
            for bi in range(batch_size):
                orig_batch = audio[bi]
                if orig_batch is None:
                    new_audios[bi] = None
                else:
                    orig = audio[bi]
                    new_audios[bi, : orig.shape[0], : orig.shape[1], : orig.shape[2]] = orig
            dict_batch[audio_key] = new_audios
            
        txt_keys = [k for k in list(dict_batch.keys()) if "text" in k]

        if len(txt_keys) != 0:
            texts = [[d[0] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            encodings = [[d[1] for d in dict_batch[txt_key]] for txt_key in txt_keys]
            draw_text_len = len(encodings)
            flatten_encodings = [e for encoding in encodings for e in encoding]
            flatten_mlms = mlm_collator(flatten_encodings)

            for i, txt_key in enumerate(txt_keys):
                texts, encodings = (
                    [d[0] for d in dict_batch[txt_key]],
                    [d[1] for d in dict_batch[txt_key]],
                )

                mlm_ids, mlm_labels = (
                    flatten_mlms["input_ids"][batch_size * (i) : batch_size * (i + 1)],
                    flatten_mlms["labels"][batch_size * (i) : batch_size * (i + 1)],
                )

                input_ids = torch.zeros_like(mlm_ids)
                attention_mask = torch.zeros_like(mlm_ids)
                for _i, encoding in enumerate(encodings):
                    _input_ids, _attention_mask = (
                        torch.tensor(encoding["input_ids"]),
                        torch.tensor(encoding["attention_mask"]),
                    )
                    input_ids[_i, : len(_input_ids)] = _input_ids
                    attention_mask[_i, : len(_attention_mask)] = _attention_mask

                dict_batch[txt_key] = texts
                dict_batch[f"{txt_key}_ids"] = input_ids
                dict_batch[f"{txt_key}_labels"] = torch.full_like(input_ids, -100)
                dict_batch[f"{txt_key}_ids_mlm"] = mlm_ids
                dict_batch[f"{txt_key}_labels_mlm"] = mlm_labels
                dict_batch[f"{txt_key}_masks"] = attention_mask

        return dict_batch
