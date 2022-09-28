from model.data.datasets import VQADataset
from .datamodule_base import BaseDataModule


class VQADataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return VQADataset

    @property
    def dataset_cls_no_false(self):
        return VQADataset

    @property
    def dataset_name(self):
        return "vqa"
