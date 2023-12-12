import torch
import torch.utils.data as data

class TopKClassificationWrapper(data.Dataset):
    def __init__(self, dataset: data.Dataset, attack_labels, seed=0, k=1) -> None:
        super().__init__()
        self.generator = torch.Generator("cpu")
        self.generator.manual_seed(seed)

        # Pregenerate attack labels
        num_classes = len(dataset.classes)

        self.src_dataset = dataset
        self.attack_labels = attack_labels

    def __getitem__(self, index):
        image, label = self.src_dataset[index]

        return image, label, self.attack_labels[index], index
    
    def __len__(self):
        return len(self.src_dataset)