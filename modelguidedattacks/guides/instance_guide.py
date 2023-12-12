import torch
from torch import nn
from torchvision.ops import MLP

from .. import losses

class InstanceGuide(nn.Module):
    def __init__(self, model: nn.Module, optimizer=torch.optim.AdamW, loss_fn=losses.CWExtensionLoss) -> None:
        super().__init__()
        
        self.guided = True
        self.model = model
        

        for p in self.model.parameters():
            p.requires_grad_(False)

        self.loss = loss_fn()
        self.optimizer = optimizer

        self.epochs = 30
        self.mlp_iterations = 5
        self.perturbation_iterations = 5

    def surject_perturbation(self, x):
        return x

    def forward(self, x, attack_targets):
        """
        x: [B, channels, H, W]
        attack_targets: [B, K]
        """

        B = x.shape[0]
        K = attack_targets.shape[-1]
        C = self.model.num_classes()

        with torch.no_grad():
            pred_clean, feats = self.model(x, return_features=True)

        # We are assuming the clean predictions are ground truth since we make that
        # constraint on the dataset side
        attack_ground_truth = pred_clean.argmax(dim=-1) # [B]

        mlp = MLP(self.model.head_features(), 
                    [self.model.head_features()]*3 + [self.model.head_features()],
                    activation_layer=nn.GELU, inplace=None).to(x.device)

        x_perturbation = nn.Parameter(torch.randn(x.shape, 
                                                 device=x.device)*1e-3)
        
        perturbation_optimizer = self.optimizer([x_perturbation], lr=1e-1)

        mlp_optimizer = self.optimizer(mlp.parameters(), lr=1e-3)

        logits_target_best = pred_clean
        feats_target_best = feats

        with torch.enable_grad():
            for i in range(self.epochs):
                for _ in range(self.mlp_iterations):
                    torch.cuda.synchronize()

                    feature_offset = mlp(feats)
                    feats_target_pred = feature_offset + feats
                    logits_target_pred = self.model.head(feats_target_pred)
                    # logits_target_pred = pred_logits
                    pred_classes = logits_target_pred.argsort(dim=-1, descending=True) # [B, C]
                    attack_successful = (pred_classes[:, :K] == attack_targets).all(dim=-1) # [B]

                    with torch.no_grad():
                        logits_target_best = torch.where(
                            attack_successful[:, None].expand(-1, C),
                            logits_target_pred,
                            logits_target_best
                        )

                        feats_target_best = torch.where(
                            attack_successful[:, None].expand(-1, self.model.head_features()),
                            feats_target_pred,
                            feats_target_best
                        )

                    mlp_loss = self.loss(logits_pred=logits_target_pred, 
                                         prediction_feats=feats_target_pred,
                                         attack_targets=attack_targets, 
                                         attack_ground_truth=attack_ground_truth, 
                                         model=self.model)
                    mlp_loss = mlp_loss.mean() + feature_offset.view(B, -1).norm(dim=-1, p=2)*1

                    mlp_optimizer.zero_grad()
                    mlp_loss.backward()
                    mlp_optimizer.step()

                feats_target_best = feats_target_best.detach()

                for _ in range(self.perturbation_iterations):
                    x_perturbed = x + self.surject_perturbation(x_perturbation)
                    prediction, perturbed_feats = self.model(x_perturbed, return_features=True)
                    pred_classes = prediction.argsort(dim=-1, descending=True) # [B, C]
                    attack_successful = (pred_classes[:, :K] == attack_targets).all(dim=-1) # [B]

                    perturbation_loss = (prediction - logits_target_best).view(B, -1).norm(dim=-1).mean()

                    perturbation_optimizer.zero_grad()
                    perturbation_loss.backward()
                    perturbation_optimizer.step()
            
        return prediction