import torch
import torchvision
import torchvision.transforms as T
from mmpretrain import datasets as mmdatasets
from mmpretrain.registry import TRANSFORMS
from mmengine.dataset import Compose

from torch import nn
from torch.utils.data import Dataset as TorchDataset

# This holds dataset instantiation functions by (dataset_name) tuple keys
DATASET_REGISTRY = {}
DATASET_PATH = "./datasets"

class MMPretrainWrapper(TorchDataset):
    def __init__(self, mmdataset) -> None:
        super().__init__()
        self.mmdataset = mmdataset

        test_pipeline = [
            dict(type='LoadImageFromFile'),
            dict(type='ResizeEdge', scale=256, edge='short'),
            dict(type='CenterCrop', crop_size=224),
            dict(type='PackInputs'),
        ]

        self.pipeline = self.init_pipeline(test_pipeline)

    def init_pipeline(self, pipeline_cfg):
        pipeline = Compose(
            [TRANSFORMS.build(t) for t in pipeline_cfg])
        return pipeline

    @property
    def classes(self):
        return self.mmdataset.CLASSES
    
    def __len__(self):
        return len(self.mmdataset)

    def __getitem__(self, index):
        sample = self.mmdataset[index]
        sample = self.pipeline(sample)

        # Our interface expects images in [0-1]
        img = sample["inputs"].float() / 255

        return img, sample["data_samples"].gt_label.item()


def register_torchvision_dataset(dataset_name, dataset_cls, dataset_kwargs_train={}, dataset_kwargs_val={}):
    def instantiate_dataset():
        train_data = dataset_cls(
            root=DATASET_PATH,
            train=True,
            download=True,
            transform=T.ToTensor()
        )

        val_data = dataset_cls(
            root=DATASET_PATH,
            train=False,
            download=True,
            transform=T.ToTensor()
        )

        return train_data, val_data

    DATASET_REGISTRY[dataset_name] = instantiate_dataset

def register_mmpretrain_dataset(dataset_name, dataset_cls, dataset_kwargs_train={}, dataset_kwargs_val={}):
    def instantiate_dataset():
        train_data = dataset_cls(**dataset_kwargs_train)
        val_data = dataset_cls(**dataset_kwargs_val)

        train_data = MMPretrainWrapper(train_data)
        val_data = MMPretrainWrapper(val_data)

        return train_data, val_data
    
    DATASET_REGISTRY[dataset_name] = instantiate_dataset

def register_default_datasets():
    register_torchvision_dataset("cifar10", torchvision.datasets.CIFAR10)
    register_torchvision_dataset("cifar100", torchvision.datasets.CIFAR100)
    register_mmpretrain_dataset("imagenet", mmdatasets.ImageNet, 
                                dataset_kwargs_train=dict(
                                    data_root = "data/imagenet", 
                                    data_prefix = "val", 
                                    ann_file = "meta/val.txt"
                                ),
                                dataset_kwargs_val=dict(
                                    data_root = "data/imagenet", 
                                    data_prefix = "val", 
                                    ann_file = "meta/val.txt"
                                ))

def get_dataset(dataset_name):
    """
    Returns an instance of a dataset

    dataset_name: Name of desired dataset
    """

    if dataset_name not in DATASET_REGISTRY:
        raise Exception("Requested dataset not in registry")        
    
    return DATASET_REGISTRY[dataset_name]()
