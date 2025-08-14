from .modelnet40 import ModelNet40
from .scanobjectnn import ScanObjectNN

dataset_list = {
    "modelnet40": ModelNet40,
    "scanobjectnn": ScanObjectNN,
}


def build_dataset(dataset, root_path, shots, allowed_classes=None):
    return dataset_list[dataset](root_path, shots, allowed_classes)
