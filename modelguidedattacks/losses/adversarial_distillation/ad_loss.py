import torch
from torch import nn
from .adversarial_distribution import AD_Distribution

class AdversarialDistillationLoss(nn.Module):
    def __init__(self, confidence=0, alpha=10, beta=0.3):
        super().__init__()

        self.alpha = alpha
        self.beta = beta

        self.distri_generator = AD_Distribution(simi_name='glove', 
                                                alpha=self.alpha, beta=self.beta)
        
        self.kl = nn.KLDivLoss(reduction='none')
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def precompute(self, attack_targets, gt_labels, config): 
        device = attack_targets.device 

        target_distribution = self.distri_generator.generate_distribution(gt_labels.cpu(), attack_targets.cpu())
        target_distribution = torch.from_numpy(target_distribution).float().to(device)

        K = attack_targets.shape[-1]
        target_distribution_topk = target_distribution.argsort(dim=-1, descending=True)[:, :K]

        assert (target_distribution_topk == attack_targets).all()

        return {
            "ad_distribution": target_distribution
        }
    
    def forward(self, logits_pred, feats_pred, feats_pred_0, attack_targets, model, ad_distribution, **kwargs):
        log_logits = self.logsoftmax(logits_pred)
        loss_kl = self.kl(log_logits, ad_distribution)
        loss_kl = torch.sum(loss_kl, dim = -1)

        return loss_kl