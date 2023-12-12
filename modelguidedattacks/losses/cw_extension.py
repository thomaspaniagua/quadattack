import torch
from torch import nn
import torch.nn.functional as F
CARLINI_COEFF_UPPER = 1e10

class CWExtensionLoss(nn.Module):
    def __init__(self, confidence=0):
        super().__init__()
        self.confidence = confidence

    def precompute(self, *args, **kwargs):
        return {}

    def forward(self, logits_pred, attack_targets,  **kwargs):
        #orign cw attack loss
        if attack_targets.dim() == 1:
            mask_logits = F.one_hot(attack_targets, logits_pred.shape[1]).float()

            real = (mask_logits * logits_pred).sum(dim=1)
            other = ((1.0 - mask_logits) * logits_pred - (mask_logits * 10000.0)
                    ).max(1)[0]
            loss_cw = torch.clamp(other - real + self.confidence, min=0.)
            return loss_cw
            
        #extended cw loss for topk attack tasks
        else:
            mask_logits = torch.zeros([logits_pred.shape[0], logits_pred.shape[1]], device=logits_pred.device)
            min_values = torch.ones(attack_targets.shape[0], dtype=torch.float, device=logits_pred.device) * 1e10
            loss_cw_topk = 0

            for i in range(attack_targets.shape[1]):
                other = ((1.0 - mask_logits) * logits_pred - (mask_logits * 10000.0)
                    ).max(1)[0]


                loss_cw_topk += torch.clamp(other - min_values + self.confidence, min=0.)
                mask_logits[torch.arange(len(attack_targets)), attack_targets[:,i]] = 1
                min_values = torch.min(logits_pred[torch.arange(len(attack_targets)), attack_targets[:,i]], min_values)

            real =  min_values
            other = ((1.0 - mask_logits) * logits_pred - (mask_logits * 10000.0)
                    ).max(1)[0]
            loss_cw_topk += torch.clamp(other - real + self.confidence, min=0.)
            constant = attack_targets.shape[1]

            return (loss_cw_topk / constant)