import logging
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from modelguidedattacks.data import get_dataset
from . import get_model

from .registry import ClsModel
from typing import Optional, List

DATASET_METADATA_DIR = "./dataset_metadata"

def correct_subset_cache_path(dataset_name: str, model_name: str, train: bool):
    filename_train_val = "train" if train else "val"
    subset_cache_filename = f"{dataset_name}_{model_name}_{filename_train_val}.p"
    subset_cache_path = os.path.join(DATASET_METADATA_DIR, subset_cache_filename)

    return subset_cache_path

@torch.no_grad()
def get_correct_subset(model: Optional[ClsModel]=None, dataset_name: Optional[str]=None, 
                       model_name: Optional[str]=None, train=True, batch_size=256, 
                       force_cache=False, device="cuda"):
    """
    model: Model to evaluate
    dataset_name: Name of dataset (not needed if model is provided)
    model_name: Name of model (not needed if model is provided)
    train: Use training dataset
    batch_size: Batch size to use while evaluating
    force_cache: Only read from cache and fail if not available

    Returns indices in dataset of correctly classified items
    """

    if model is not None:
        assert dataset_name is None
        assert model_name is None

    if dataset_name is not None or model_name is not None:
        assert dataset_name is not None
        assert model_name is not None
        assert model is None

    if dataset_name is None:
        dataset_name = model.dataset_name

    if model_name is None:
        model_name = model.model_name

    filename_train_val = "train" if train else "val"
    subset_cache_filename = f"{dataset_name}_{model_name}_{filename_train_val}.p"
    subset_cache_path = os.path.join(DATASET_METADATA_DIR, subset_cache_filename)

    os.makedirs(DATASET_METADATA_DIR, exist_ok=True)

    if os.path.exists(subset_cache_path):
        correct_subset = torch.load(subset_cache_path)
        return correct_subset

    if force_cache:
        raise Exception("Cache not found and requested for cached correct subset.")

    logging.info(f"No cache found. Computing correct subset for {dataset_name}-{model_name} Train: {train}")

    device = device if model is None else model.device    

    if model is None:
        model = get_model(dataset_name, model_name, device)
    
    model.eval()

    train_dataset, val_dataset = get_dataset(dataset_name)

    dataset = train_dataset
    
    if not train:
        dataset = val_dataset

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    correct_indices = []

    for batch_i, (batch_imgs, batch_gt_class) in tqdm(enumerate(dataloader), total=len(dataloader)):
        if torch.device(model.device).type.startswith("cuda"):
            torch.cuda.synchronize(model.device)

        data_start_index = batch_i * batch_size
        predictions = model(batch_imgs.to(model.device)) # [B, C]
        prediction_class_idx = predictions.argmax(dim=-1) # [B] (long)
        prediction_correct = prediction_class_idx == batch_gt_class.to(model.device)
        batch_correct_idxs = data_start_index + prediction_correct.nonzero()[:, 0]
        batch_correct_idxs = batch_correct_idxs.tolist()

        correct_indices.extend(batch_correct_idxs)

    correct_subset = set(correct_indices)
    torch.save(correct_subset, subset_cache_path)

    return set(correct_indices)

def get_correct_subset_for_models(model_names: List[str], dataset_name, device, train):
    correct_intersection = None
    for model_name in model_names:
        model_correct_subset = get_correct_subset(model_name=model_name, dataset_name=dataset_name,
                                                   device=device, train=train)
        
        if correct_intersection is None:
            correct_intersection = model_correct_subset
        else:
            correct_intersection = model_correct_subset.intersection(correct_intersection)

    return list(correct_intersection)