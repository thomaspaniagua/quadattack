from typing import Any
import os
import torch
import ignite.distributed as idist
import torchvision
import torchvision.transforms as T
from torch.utils import data as torch_data

from .classification_wrapper import TopKClassificationWrapper
from torch.utils.data import Subset
from modelguidedattacks.data import get_dataset
from modelguidedattacks.cls_models.accuracy import get_correct_subset_for_models, DATASET_METADATA_DIR

from tqdm import tqdm

def get_gt_labels(dataset: TopKClassificationWrapper, train:bool, dataset_name:str):
    training_str = "train" if train else "val"
    save_name = os.path.join(DATASET_METADATA_DIR, f"{dataset_name}_labels_{training_str}.p")

    if os.path.exists(save_name):
        print ("Found labels cache")
        return torch.load(save_name)

    dataloader = torch_data.DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)

    gt_labels = []

    for batch in tqdm(dataloader):
        gt_labels.extend(batch[1].tolist())

    gt_labels = torch.tensor(gt_labels)

    torch.save(gt_labels, save_name)

    return gt_labels

def class_balanced_sampling(dataset, gt_labels: torch.Tensor,
                            correct_labels: list, total_samples=1000):
    num_classes = len(dataset.classes)

    correct_labels = torch.tensor(correct_labels)
    correct_mask = torch.zeros((len(dataset), ), dtype=torch.bool)
    correct_mask[correct_labels] = True

    sampled_indices = 0

    total_sampled_indices = 0
    sampled_indices = [[] for i in range(num_classes)]

    shuffled_inds = torch.randperm(len(dataset))

    for sample_cnt, sample_i in enumerate(shuffled_inds):
        if not correct_mask[sample_i]:
            continue

        sample_class = gt_labels[sample_i]
        desired_samples_in_class = (total_sampled_indices // num_classes) + 1

        if len(sampled_indices[sample_class]) < desired_samples_in_class:
            sampled_indices[sample_class].append(sample_i.item())
            total_sampled_indices += 1

            if total_sampled_indices >= total_samples:
                break

    flattened_indices = []
    for class_samples in sampled_indices:
        flattened_indices.extend(class_samples)

    return torch.tensor(flattened_indices)

def sample_attack_labels(dataset, gt_labels, k, sampler):
    """
    dataset: Dataset we're generating attack labels for
    gt_labels: List of gt idx for each sample in a dataset
    k: attack size
    sampler: ["random"]
    """

    # Sample from uniform and argsort to simulate
    # a batched randperm
    attack_label_uniforms = torch.rand((len(gt_labels), len(dataset.classes)))

    # We don't want to sample the gt class for any samples
    batch_inds = torch.arange(len(gt_labels))
    attack_label_uniforms[batch_inds, gt_labels] = -1.

    attack_labels = attack_label_uniforms.argsort(dim=-1, descending=True)[:, :k]

    return attack_labels

def setup_data(config: Any, rank):
    """Download datasets and create dataloaders

    Parameters
    ----------
    config: needs to contain `data_path`, `train_batch_size`, `eval_batch_size`, and `num_workers`
    """

    dataset_train, dataset_eval = get_dataset(config.dataset)

    train_subset = None
    val_subset = None

    attack_labels_train = None
    attack_labels_val = None

    if rank == 0:
        gt_labels_train = get_gt_labels(dataset_train, True, config.dataset)
        gt_labels_val = get_gt_labels(dataset_eval, False, config.dataset)

        attack_labels_train = sample_attack_labels(dataset_train, gt_labels_train, k=config.k,
                                                   sampler=config.attack_sampling)
        attack_labels_val = sample_attack_labels(dataset_eval, gt_labels_val, k=config.k,
                                                 sampler=config.attack_sampling)
    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        correct_train_set = get_correct_subset_for_models(config.compare_models, 
                                                        config.dataset, device, 
                                                        train=True)
        
        correct_eval_set = get_correct_subset_for_models(config.compare_models, 
                                                        config.dataset, device, 
                                                        train=False)
        
        # Balanced sampling
        train_subset = class_balanced_sampling(dataset_train, gt_labels_train,
                                               correct_train_set)
        
        val_subset = class_balanced_sampling(dataset_eval, gt_labels_val,
                                             correct_eval_set)
        
        if config.overfit:
            rand_inds = torch.randperm(len(val_subset))[:16]
            train_subset = train_subset[rand_inds]
            val_subset = val_subset[rand_inds]
    
    train_subset = idist.broadcast(train_subset, safe_mode=True)
    val_subset = idist.broadcast(val_subset, safe_mode=True)

    attack_labels_train = idist.broadcast(attack_labels_train, safe_mode=True)
    attack_labels_val = idist.broadcast(attack_labels_val, safe_mode=True)

    dataset_train = TopKClassificationWrapper(dataset_train, k=config.k, 
                                              attack_labels=attack_labels_train)
    dataset_eval = TopKClassificationWrapper(dataset_eval, k=config.k, 
                                             attack_labels=attack_labels_val)

    dataset_train = Subset(dataset_train, train_subset)
    dataset_eval = Subset(dataset_eval, val_subset)

    # if config.overfit:
    #     dataset_train = Subset(dataset_train, range(2))
    #     dataset_eval = dataset_train
    # else:
    #     dataset_eval = Subset(dataset_eval, torch.randperm(len(dataset_eval))[:1000].tolist() )

    dataloader_train = idist.auto_dataloader(
        dataset_train,
        batch_size=config.train_batch_size,
        shuffle=not config.overfit,
        num_workers=config.num_workers,
    )
    dataloader_eval = idist.auto_dataloader(
        dataset_eval,
        batch_size=config.eval_batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )
    return dataloader_train, dataloader_eval
