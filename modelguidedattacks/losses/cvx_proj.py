import torch
from torch import nn
import torch.nn.functional as F
import cvxpy as cp
import qpth
from . import _qp_solver_patch

def solve_qp(Q, P, G, H):
    B = Q.shape[0]

    if B == 1:
        # Batch size of 1 has weird instabilities
        # I imagine there is a .squeeze() or something inside the QP solver's code
        # that messes up broadcasting dimensions when batch dimension is 1 so let's
        # artificially make 2 solutions when we need 1

        Q = Q.expand(2, -1, -1)
        P = P.expand(2, -1)
        G = G.expand(2, -1, -1)
        H = H.expand(2, -1)

    e = torch.empty(0, device=Q.device)
    z_sol = qpth.qp.QPFunction(verbose=-1, eps=1e-2, check_Q_spd=False)(Q, P, G, H, e, e)

    if B == 1:
        z_sol = z_sol[:1]

    return z_sol

class CVXProjLoss(nn.Module):
    def __init__(self, confidence=0):
        super().__init__()
        self.confidence = confidence

    def precompute(self, attack_targets, gt_labels, config):
        return {
            "margin": config.cvx_proj_margin
        }

    def forward(self, logits_pred, feats_pred, feats_pred_0, attack_targets, model, margin, **kwargs):
        device = logits_pred.device
        head_W, head_bias = model.head_matrices()

        num_feats = head_W.shape[1]
        num_classes = head_W.shape[0]

        K = attack_targets.shape[-1]
        B = logits_pred.shape[0]
        
        # Start with all classes should be less than smallest attack target
        D = -torch.eye(num_classes, device=device)[None].repeat(B, 1, 1) # [B, C, C]
        attack_targets_write = attack_targets[:, -1][:, None, None].expand(-1, D.shape[1], -1)
        D.scatter_(dim=2, index=attack_targets_write, src=torch.ones(attack_targets_write.shape, device=device))

        # Clear out the constraint row for each item in the attack targets
        attack_targets_clear = attack_targets[:, :, None].expand(-1, -1, D.shape[-1])
        D.scatter_(dim=1, index=attack_targets_clear, src=torch.zeros(attack_targets_clear.shape, device=device))

        batch_inds = torch.arange(B, device=device)[:, None].expand(-1, K - 1)
        attack_targets_pos = attack_targets[:, :-1] # [B, K-1]
        attack_targets_neg = attack_targets[:, 1:] # [B, K-1]

        attack_targets_neg_inds = torch.stack((
            batch_inds,
            attack_targets_neg,
            attack_targets_neg
        ), dim=0) # [3, B, K - 1]
        attack_targets_neg_inds = attack_targets_neg_inds.view(3, -1)

        D[attack_targets_neg_inds[0], attack_targets_neg_inds[1], attack_targets_neg_inds[2]] = -1

        attack_targets_pos_inds = torch.stack((
            batch_inds,
            attack_targets_neg,
            attack_targets_pos
        ), dim=0) # [3, B, K - 1]

        D[attack_targets_pos_inds[0], attack_targets_pos_inds[1], attack_targets_pos_inds[2]] = 1

        A = head_W
        b = head_bias

        Q = 2*torch.eye(feats_pred.shape[1], device=device)[None].expand(B, -1, -1)

        # We want the solution features to be as close as possible
        # to the current features but also head on the direction of 
        # the smallest possible perturbation from the initial predicted
        # features
        anchor_feats = feats_pred

        P = -2*anchor_feats.expand(B, -1)

        G = -D@A
        H = -(margin - D @ b)

        # Constraints are indexed by smaller logit
        # First attack target isn't smaller than any logit, so its 
        # constraint index is redundant, but we keep it for easier parallelization
        # Make this constraint all 0s
        zero_inds = attack_targets[:, 0:1] # [B, 1]
        H.scatter_(dim=1, index=zero_inds, src=torch.zeros(zero_inds.shape, device=device))

        z_sol = solve_qp(Q, P, G, H)

        loss = (feats_pred - z_sol).square().sum(dim=-1)

        # loss_check = self.forward_check(logits_pred, feats_pred, attack_targets, model, **kwargs)
        return loss