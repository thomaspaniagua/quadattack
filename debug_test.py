import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from torch.nn import functional as F
import torch
from modelguidedattacks import cls_models
import time

torch.manual_seed(0)
device = "cuda"
# model = cls_models.get_model("imagenet", "resnet18", device)

rand_feats = torch.randn(1, 512, device=device)
attack_targets = [4, 7, 5, 9, 2]

# # pred_logits = model.head(rand_feats)

# # head_W, head_bias = model.head_matrices()

(head_W, head_bias, pred_logits) = torch.load("debugsaveimagenet.save")

rand_feats, rand_logits, attack_targets = torch.load("attack_case.p", map_location=device)
reconstructed_logits = rand_feats@head_W.T + head_bias

num_feats = head_W.shape[1]
num_classes = head_W.shape[0]
x = cp.Variable(num_feats)

anchor_feats = cp.Parameter(x.shape)
A = cp.Parameter(head_W.shape)
b = cp.Parameter(head_bias.shape)

logits = A@x + b

MARGIN = 0.1

# constraints  = []
# for i in range(len(attack_targets) - 1):
#     constraints.append( logits[attack_targets[i]] - logits[attack_targets[i+1]] >= MARGIN)

# for i in range(num_classes):
#     if i in attack_targets:
#         continue

#     constraints.append(logits[attack_targets[-1]] - logits[i] >= MARGIN )

# objective = cp.Minimize(0.5 * cp.pnorm(x - anchor_feats, p=2))
# problem = cp.Problem(objective, constraints)

# anchor_feats.value = rand_feats[0].cpu().numpy()
# A.value = head_W.detach().cpu().numpy()
# b.value = head_bias.detach().cpu().numpy()

# start_time = time.time()
# problem.solve()
# print ("Non vectorized sol", time.time() - start_time)

# logits_sol_torch = torch.from_numpy(logits.value)
# logits_check = logits_sol_torch.argsort(descending=True)

# feats_sol = torch.from_numpy(x.value[:, None]).float().to(rand_feats)
# sol_feat_norm = (feats_sol[:, 0].cpu() - rand_feats[0].cpu()).norm(dim=-1)
# sol_logits = head_W@feats_sol + head_bias[:, None]
# sol_sort = sol_logits.argsort(dim=0, descending=True)


# Constraint matrix
num_constraints = num_classes - 1
D = torch.zeros((num_classes), num_constraints)

non_attack_targets = list(set(range(num_classes)) - set(attack_targets))

for constraint_cursor in range(num_constraints):
    if constraint_cursor < len(attack_targets) - 1:
        D[attack_targets[constraint_cursor], constraint_cursor] = 1
        D[attack_targets[constraint_cursor + 1], constraint_cursor] = -1
    else:
        non_attack_i = constraint_cursor - len(attack_targets) + 1
        D[attack_targets[-1], constraint_cursor] = 1
        D[non_attack_targets[non_attack_i], constraint_cursor] = -1

D = D.T
# vectorized_differences = D @ logits
# vectorized_constraint = vectorized_differences >= torch.full(vectorized_differences.shape, fill_value=MARGIN).numpy()

# Q = 2*torch.eye(x.shape[0]).numpy()
# P = -2*anchor_feats

# G = D@A
# H = MARGIN - D @ b

# G = -G
# H = -H

# vectorized_constraint = G@x <= H

# objective = cp.Minimize((1/2)*cp.quad_form(x, Q) + P.T@x)
# problem = cp.Problem(objective, [vectorized_constraint])

# anchor_feats.value = rand_feats[0].cpu().numpy()
# A.value = head_W.detach().cpu().numpy()
# b.value = head_bias.detach().cpu().numpy()

# start_time = time.time()
# problem.solve()
# print ("vectorized sol", time.time() - start_time)

# logits_sol_torch = torch.from_numpy(logits.value)
# logits_check = logits_sol_torch.argsort(descending=True)
# feats_sol = torch.from_numpy(x.value[:, None]).float().to(rand_feats)
# sol_feat_norm = (feats_sol[:, 0].cpu() - rand_feats[0].cpu()).norm(dim=-1)
# sol_logits = head_W@feats_sol + head_bias[:, None]
# sol_sort = sol_logits.argsort(dim=0, descending=True)

import qpth


B = 2
nz = num_feats
nineq = num_constraints
device = "cuda"

attack_targets = attack_targets.expand(B, -1)
K = attack_targets.shape[-1]

# Start with all classes should be less than smallest attack target
D = -torch.eye(num_classes, device=device)[None].repeat(B, 1, 1)
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

A = head_W.detach().to(device)
b = head_bias.detach().to(device)
D = D.to(device)

#rand_feats: [B, num_features]
Q = 2*torch.eye(nz, device=device)[None].expand(B, -1, -1)
P = -2*rand_feats.to(device).expand(B, -1)

# G = torch.randn(B, nineq, nz, device=device)
G = -D@A

# h = torch.randn(B, nineq)
H = -(MARGIN - D @ b)

# Constraints are indexed by smaller logit
# First attack target isn't smaller than any logit, so its 
# constraint index is redundant, but we keep it for easier parallelization
# Make this constraint all 0s
zero_inds = attack_targets[:, 0:1] # [B, 1]
H.scatter_(dim=1, index=zero_inds, src=torch.zeros(zero_inds.shape, device=device))

e = torch.empty(0, device=device)

Q_t, P_t, G_t, H_t = torch.load("qpinputs.p", map_location=device)

z_sol = qpth.qp.QPFunction(verbose=True, check_Q_spd=False)(Q, P, G, H, e, e).T

logits = A@z_sol + b[:, None]

x = 5