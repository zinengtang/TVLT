from model.data.datasets import MOSEIEMODataset
from .datamodule_base import BaseDataModule


class MOSEIEMODataModule(BaseDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def dataset_cls(self):
        return MOSEIEMODataset

    @property
    def dataset_cls_no_false(self):
        return MOSEIEMODataset

    @property
    def dataset_name(self):
        return "moseiemo"
