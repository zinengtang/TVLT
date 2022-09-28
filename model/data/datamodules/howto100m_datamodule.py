from model.data.datasets import Howto100mDataset
from .datamodule_base import BaseDataModule


class Howto100mDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return Howto100mDataset

    @property
    def dataset_cls_no_false(self):
        return Howto100mDataset

    @property
    def dataset_name(self):
        return "howto100m"
