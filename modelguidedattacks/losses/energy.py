import torch
from torch import nn
from ignite.metrics import Loss
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from typing import Callable, cast, Dict, Sequence, Tuple, Union

def get_correct_mask(y_pred, y_attack):
    k = y_attack.shape[-1]

    y_pred_indices = y_pred.argsort(dim=-1, descending=True) # [N, C]

    correct = (y_pred_indices[:, :k] == y_attack).all(dim=-1)
    return correct

class EnergyLoss(Loss):
    def __init__(self, loss_fn, reduction="mean", device = ...):
        super().__init__(loss_fn, device=device)
        self.reduction = reduction

    @reinit__is_reduced
    def reset(self) -> None:
        self._sum = torch.tensor(0.0, device=self._device)
        self._min = torch.tensor(torch.inf, device=self._device)
        self._max = torch.tensor(0.0, device=self._device)
        self._num_examples = 0

    @reinit__is_reduced
    def update(self, output: Sequence[Union[torch.Tensor, Dict]]) -> None:
        if len(output) == 2:
            y_pred, y = cast(Tuple[torch.Tensor, torch.Tensor], output)
            kwargs: Dict = {}
        else:
            y_pred, y, kwargs = cast(Tuple[torch.Tensor, torch.Tensor, Dict], output)

        sample_energies = self._loss_fn(y_pred, y, **kwargs).detach()

        n = len(sample_energies)

        if n > 0:
            self._sum += sample_energies.sum()
            self._min = torch.minimum(self._min, sample_energies.min())
            self._max = torch.maximum(self._max, sample_energies.max())
            self._num_examples += n

    @sync_all_reduce("_sum", "_num_examples", "_min:MIN", "_max:MAX")
    def compute(self) -> float:

        if self.reduction == "mean":
            if self._num_examples == 0:
                return torch.inf
            
            return self._sum.item() / self._num_examples
        elif self.reduction == "max":
            if self._num_examples == 0:
                return torch.nan
    
            return self._max.item()
        elif self.reduction == "min":
            if self._num_examples == 0:
                return torch.inf
            
            return self._min.item()
        else:
            assert False

class Energy(nn.Module):
    def __init__(self, p="2") -> None:
        super().__init__()
        self.p = p

    def forward(self, y_pred, y_attack, perturbations, **kwargs):
        correct = get_correct_mask(y_pred, y_attack)

        # Don't want to take into account perturbations of
        # unsuccessful attacks

        perturbations = perturbations[correct]
        perturbations = perturbations.flatten(1)

        return torch.linalg.vector_norm(perturbations, dim=-1, ord=self.p)