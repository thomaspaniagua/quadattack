import torch
from ignite.metrics import Accuracy, Loss
from typing import Sequence

class TopKAccuracy(Accuracy):
    def update(self, output: Sequence[torch.Tensor], **kwargs) -> None:
        y_pred, y_attack = output[0].detach(), output[1].detach()
        k = y_attack.shape[-1]

        y_pred_indices = y_pred.argsort(dim=-1, descending=True) # [N, C]

        correct = (y_pred_indices[:, :k] == y_attack).all(dim=-1)

        self._num_correct += torch.sum(correct).to(self._device)
        self._num_examples += correct.shape[0]