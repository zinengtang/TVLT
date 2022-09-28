from model.data.datasets import MsrvttDataset
from .datamodule_base import BaseDataModule


class MsrvttDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MsrvttDataset

    @property
    def dataset_cls_no_false(self):
        return MsrvttDataset

    @property
    def dataset_name(self):
        return "msrvtt"
