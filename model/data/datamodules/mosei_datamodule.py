from model.data.datasets import MOSEIDataset
from .datamodule_base import BaseDataModule


class MOSEIDataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MOSEIDataset

    @property
    def dataset_cls_no_false(self):
        return MOSEIDataset

    @property
    def dataset_name(self):
        return "mosei"
