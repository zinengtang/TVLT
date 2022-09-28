import torch

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizer,
)

def get_pretrained_tokenizer(from_pretrained):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            BertTokenizer.from_pretrained(
                from_pretrained, do_lower_case="uncased" in from_pretrained
            )
        torch.distributed.barrier()
    return BertTokenizer.from_pretrained(
        from_pretrained, do_lower_case="uncased" in from_pretrained
    )


class BaseDataModule(LightningDataModule):
    def __init__(self, _config):
        super().__init__()

        self.data_dir = _config["data_root"]

        self.num_workers = _config["num_workers"]
        self.batch_size = _config["per_gpu_batchsize"]
        self.eval_batch_size = self.batch_size
        
        self.video_size = _config["video_size"]
        self.audio_size = _config["audio_size"]
        self.max_text_len = _config["max_text_len"]
        self.num_frames = _config["num_frames"]
        self.draw_false_audio = _config["draw_false_audio"]
        self.draw_false_video = _config["draw_false_video"]
        self.draw_false_text = _config["draw_false_text"]
        self.audio_only = _config["audio_only"]
        self.video_only = _config["video_only"]
        self.use_audio = _config["use_audio"]
        self.use_text = _config["use_text"]


        tokenizer = _config["tokenizer"]
        self.tokenizer = get_pretrained_tokenizer(tokenizer)
        self.vocab_size = self.tokenizer.vocab_size

        collator = (
            DataCollatorForWholeWordMask
            if _config["whole_word_masking"]
            else DataCollatorForLanguageModeling
        )

        self.mlm_collator = collator(
            tokenizer=self.tokenizer, mlm=True, mlm_probability=_config["mlm_prob"]
        )
        self.setup_flag = False

    @property
    def dataset_cls(self):
        raise NotImplementedError("return tuple of dataset class")

    @property
    def dataset_name(self):
        raise NotImplementedError("return name of dataset")

    def set_train_dataset(self):
        self.train_dataset = self.dataset_cls(
            self.data_dir,
            split="train",
            video_size=self.video_size,
            audio_size=self.audio_size,
            max_text_len=self.max_text_len,
            num_frames=self.num_frames,
            draw_false_audio=self.draw_false_audio,
            draw_false_video=self.draw_false_video,
            draw_false_text=self.draw_false_text,
            audio_only=self.audio_only,
            video_only=self.video_only,
            use_audio=self.use_audio,
            use_text=self.use_text,
        )

    def set_val_dataset(self):
        self.val_dataset = self.dataset_cls(
            self.data_dir,
            split="val",
            video_size=self.video_size,
            audio_size=self.audio_size,
            max_text_len=self.max_text_len,
            num_frames=self.num_frames,
            draw_false_audio=self.draw_false_audio,
            draw_false_video=self.draw_false_video,
            draw_false_text=self.draw_false_text,
            audio_only=self.audio_only,
            video_only=self.video_only,
            use_audio=self.use_audio,
            use_text=self.use_text,
        )

        if hasattr(self, "dataset_cls_no_false"):
            self.val_dataset_no_false = self.dataset_cls_no_false(
                self.data_dir,
                split="val",
                video_size=self.video_size,
                audio_size=self.audio_size,
                max_text_len=self.max_text_len,
                num_frames=self.num_frames,
                draw_false_audio=0,
                draw_false_video=0,
                draw_false_text=0,
                audio_only=self.audio_only,
                video_only=self.video_only,
                use_audio=self.use_audio,
                use_text=self.use_text,
            )

    def make_no_false_val_dset(self, image_only=False, video_only=False, audio_only=False):
        return self.dataset_cls(
            self.data_dir,
            split="val",
            video_size=self.video_size,
            audio_size=self.audio_size,
            max_text_len=self.max_text_len,
            num_frames=self.num_frames,
            draw_false_audio=0,
            draw_false_video=0,
            draw_false_text=0,
            audio_only=audio_only,
            video_only=video_only,
            use_audio=self.use_audio,
            use_text=self.use_text,
        )

    def set_test_dataset(self):
        self.test_dataset = self.dataset_cls(
            self.data_dir,
            split="test",
            video_size=self.video_size,
            audio_size=self.audio_size,
            max_text_len=self.max_text_len,
            num_frames=self.num_frames,
            draw_false_audio=self.draw_false_audio,
            draw_false_video=self.draw_false_video,
            draw_false_text=self.draw_false_text,
            audio_only=self.audio_only,
            video_only=self.video_only,
            use_audio=self.use_audio,
            use_text=self.use_text,
        )

    def setup(self, stage):
        if not self.setup_flag:
            self.set_train_dataset()
            self.set_val_dataset()
            self.set_test_dataset()

            self.train_dataset.tokenizer = self.tokenizer
            self.val_dataset.tokenizer = self.tokenizer
            self.test_dataset.tokenizer = self.tokenizer

            self.setup_flag = True

    def train_dataloader(self):
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.train_dataset.collate,
        )
        return loader

    def val_dataloader(self):
        loader = DataLoader(
            self.val_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.val_dataset.collate,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.eval_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.test_dataset.collate,
        )
        return loader
