import copy
import json
import warnings
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from transformers.optimization import AdamW
from transformers import get_cosine_schedule_with_warmup
from model.modules import heads, objectives, model_utils
import model.modules.tvlt as tvlt

from huggingface_sb3 import load_from_hub


class Transformer(pl.LightningModule):
    def __init__(self, config, model_type='transformer'):
        super().__init__()

        self.model_type = model_type
        self.learning_rate = config["learning_rate"]
        self.weight_decay = config["weight_decay"]
        self.patch_size = config["patch_size"]
        self.audio_patch_size = config["audio_patch_size"]
        self.warmup_steps = config["warmup_steps"]
        
        self.transformer = getattr(tvlt, config["model_type"])(config=config)
        
        self.save_hyperparameters()
        model_utils.set_metrics(self)
        self.current_tasks = list()
        self.apply(objectives.init_weights)
        self.transformer.init_weights()

        # ===================== load checkpoints ======================

        if config["load_local_path"]:
            state_dict = torch.load(config["load_local_path"], map_location="cpu")
            if "model" in state_dict.keys():
                state_dict = state_dict["model"]         
            elif "state_dict" in state_dict.keys():
                state_dict = state_dict['state_dict']
            self.load_state_dict(state_dict, strict=config['strict_load'])
            
        if config["load_hub_path"]:
            ckpt_path = load_from_hub(repo_id="TVLT/models", filename=config["load_hub_path"])
            self.transformer.load_state_dict(torch.load(ckpt_path), strict=config['strict_load'])
            
            
    def infer(
        self,
        batch,
        audio_embeds=None,
        audio_masks=None,
        video_embeds=None,
        video_masks=None,
        audio_token_type_idx=1,
        video_token_type_idx=2,
        mask_text=False,
        mask_visual=False,
        use_mae=False
    ):
        
        do_mlm = "_mlm" if mask_text else ""        
        videokey = "video_data"
        audiokey = "audio_data"
        textkey = "text_ids"+do_mlm
        
        use_audio = audiokey in list(batch.keys())
        use_video = videokey in list(batch.keys())                
        has_text = textkey in list(batch.keys())
        
        if has_text:    
            text_ids = batch[f"text_ids{do_mlm}"]
            text_labels = batch[f"text_labels{do_mlm}"]
            text_masks = batch[f"text_masks"]
            text_labels_mlm = batch[f"text_labels_mlm"] if f"text_labels_mlm" in batch.keys() and mask_text else None
        else:
            text_ids = None
            text_labels = None
            text_masks = None 
            text_embeds = None
            text_labels_mlm = None
            
        if use_audio:
            audio = batch[audiokey]
        else:
            audio = None

        if use_video:
            video = batch[videokey] 
        else:
            video = None
                      
        text_feats, audio_feats, video_feats = None, None, None
        audio_labels_mlm = video_labels_mlm = None


        cls_feats, audio_feats, video_feats, text_feats, audio_masks, video_masks = self.transformer(text_ids=text_ids, text_masks=text_masks, audio=audio, audio_masks=audio_masks, video=video, video_masks=video_masks, mask_visual=mask_visual, use_mae=use_mae)

        ret = {
            "text_feats": text_feats,
            "audio_feats": audio_feats,
            "video_feats": video_feats,
            "text_feats": text_feats,
            "cls_feats": cls_feats,
            "video_masks": video_masks,
            "video": video,
            "audio_masks": audio_masks,
            "audio": audio,
            "text_labels": text_labels,
            "text_ids": text_ids,
            "text_masks": text_masks,
        }

        return ret

    def forward(self, batch):
        ret = dict()
        if len(self.current_tasks) == 0:
            ret.update(self.infer(batch))
            return ret

        # Video Audio Matching
        if "vam" in self.current_tasks:
            ret.update(objectives.compute_vam(self, batch))
            
        # Video Audio Retrieval
        if "vatr" in self.current_tasks:
            ret.update(objectives.compute_vatr(self, batch))    
            
        # Video Text Matching
        if "vtm" in self.current_tasks:
            ret.update(objectives.compute_vtm(self, batch))
            
        # Masked Language Modeling
        if "mlm" in self.current_tasks:
            ret.update(objectives.compute_mlm(self, batch))

        if "mae_audio" in self.current_tasks and "mae_video" in self.current_tasks:
            ret.update(objectives.compute_mae_joint(self, batch, self.patch_size, self.audio_patch_size))       
        
        # Masked Patch Prediction
        elif "mae_audio" in self.current_tasks:
            ret.update(objectives.compute_mae_audio(self, batch, self.audio_patch_size))
            
        elif "mae_video" in self.current_tasks:
            ret.update(objectives.compute_mae_video(self, batch, self.patch_size))
       
        # Visual Question Answering
        if "vqa" in self.current_tasks:
            ret.update(objectives.compute_vqa(self, batch))
            
        if "mosei" in self.current_tasks:
            ret.update(objectives.compute_mosei(self, batch))
            
        if "moseiemo" in self.current_tasks:
            ret.update(objectives.compute_moseiemo(self, batch))

        return ret

    def training_step(self, batch, batch_idx):    
        model_utils.set_task(self)     
        output = self(batch)
        total_loss = sum([v for k, v in output.items() if "loss" in k]) 
        return total_loss
    
    def training_epoch_end(self, outs):
        model_utils.epoch_wrapup(self)

    def validation_step(self, batch, batch_idx):
        model_utils.set_task(self)
        output = self(batch)   
                
    def validation_epoch_end(self, outs):
        model_utils.epoch_wrapup(self)

    def test_step(self, batch, batch_idx):
        model_utils.set_task(self)
        output = self(batch)
        ret = dict()

        if self.hparams.config["loss_names"]["vqa"] > 0:
            ret.update(objectives.vqa_test_step(self, batch, output))

        return ret

    def test_epoch_end(self, outs):
        model_name = self.hparams.config["load_path"].split("/")[-1][:-5]

        if self.hparams.config["loss_names"]["vqa"] > 0:
            objectives.vqa_test_wrapup(outs, model_name)
        model_utils.epoch_wrapup(self)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.learning_rate, eps=1e-8, betas=(0.9, 0.98), weight_decay=self.weight_decay)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.warmup_steps,
        )
        sched = {"scheduler": scheduler, "interval": "step"}
        return (
            [optimizer],
            [sched],
        )
