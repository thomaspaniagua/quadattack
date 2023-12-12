import torch
from torch import nn

def generalized_mean(x, p, dim):
    x_type = x.dtype
    x = x.to(torch.double)
    x = x**p
    x = x.mean(dim=dim)
    x = x**(1/p)
    return x.to(x_type)

def surject_to_positive(x, c=5):
    assert x.min() >= -1
    assert x.max() <= 1

    return c + c * x

def surject_from_positive(x, c=5):
    return (x - c) / c

class BoilerplateLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.p = 9

    def forward(self, y_pred, y_attack, **kwargs):
        y_pred = y_pred.softmax(dim=-1)

        C = y_pred.shape[1]
        K = y_attack.shape[1]
        desired_mask = torch.zeros_like(y_pred, dtype=torch.bool)
        desired_mask.scatter_(dim=1, index=y_attack, 
                              src=torch.ones_like(y_attack, dtype=torch.bool))
        
        y_not_in_attack = (~desired_mask).nonzero()[:, 1].view(-1, C - K)

        y_pred_in_attack = torch.gather(y_pred, dim=1, index=y_attack)
        y_pred_not_in_attack = torch.gather(y_pred, dim=1, index=y_not_in_attack)

        y_pred_in_attack_min = y_pred_in_attack.min(dim=-1).values #generalized_mean(y_pred_in_attack, -self.p, dim=1)
        y_pred_not_in_attack_max = y_pred_not_in_attack.max(dim=-1).values #generalized_mean(y_pred_not_in_attack, self.p, dim=1)

        macro_loss = (y_pred_not_in_attack_max - y_pred_in_attack_min)
        sorting_loss = y_pred_in_attack.diff(dim=-1)

        # Surject sorting_loss to positive domain, since it goes [-1,1] we can just shift by 1
        sorting_loss = surject_to_positive(sorting_loss)
        sorting_loss = generalized_mean(sorting_loss, p=9, dim=-1)

        # Surject back
        sorting_loss = surject_from_positive(sorting_loss)

        catted_loss = torch.stack([macro_loss, sorting_loss], dim=-1)
        catted_loss_pos = surject_to_positive(catted_loss)
        
        final_loss_pos = generalized_mean(catted_loss_pos, p=10, dim=-1)
        final_loss = surject_from_positive(final_loss_pos)

        return final_loss
