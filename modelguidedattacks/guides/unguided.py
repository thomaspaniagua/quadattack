import torch
from torch import nn
from .. import losses
import ignite.distributed as idist
import torch_optimizer 
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.nn import functional as F
import os

import shutil
from modelguidedattacks.cls_models.registry import MMPretrainVisualTransformerWrapper
from modelguidedattacks.data.imagenet_metadata import imgnet_idx_to_name

class Unguided(nn.Module):
    def __init__(self, model: nn.Module, config, optimizer=torch.optim.AdamW, seed=0, iterations=1000,
                 loss_fn=losses.CVXProjLoss, lr=1e-3, 
                 binary_search_steps=1, topk_loss_coef_upper=10.,
                 topk_loss_coef_lower=0.) -> None:
        super().__init__()

        self.guided = False
        self.model = model
        self.seed = seed
        self.iterations = iterations
        self.loss = loss_fn()
        self.optimizer = optimizer
        self.lr = lr

        self.binary_search_steps = binary_search_steps
        self.topk_loss_coef_upper = topk_loss_coef_upper
        self.topk_loss_coef_lower = topk_loss_coef_lower
        self.config = config

    def surject_perturbation(self, x, max_norm=5.):
        x_shape = x.shape

        x = x.flatten(1)
        x_norm = x.norm(dim=-1)
        x_unit = x / x_norm[:, None]

        x_norm_outside = x_norm > max_norm
        x_norm_outside = x_norm_outside.expand_as(x)

        x = torch.where(x_norm_outside, x_unit*max_norm, x)

        return x.view(x_shape)

    @torch.enable_grad()
    def attack(self, x, attack_targets, gt_labels, topk_coefs):
        """
        For a given set of topk coefficients, this function computes
        best energy attack in the given number of iterations and configuration

        x: [B, C, H, W] [0-1 for colors]
        attack_targets: [B, K] (long)
        gt_labels: [B] (long)
        topk_coefs: [B] (floats)
        """

        topk_coefs = topk_coefs.clone()
        K = attack_targets.shape[-1]

        x_perturbation = nn.Parameter(torch.randn(x.shape, 
                                                 device=x.device)*2e-3)

        with torch.no_grad():
            prediction_logits_0, prediction_feats_0 \
                = self.model(x, return_features=True)

        best_perturbations = torch.zeros_like(x) # [B, 3, H, W]
        has_successful_attack = torch.zeros(x.shape[0], dtype=torch.long, device=x.device) # [B]
        best_energy = torch.full((x.shape[0],), float('inf'), device=x.device) # [B]

        pbar = tqdm(range(self.iterations))

        for i in pbar:

            if i == self.config.opt_warmup_its:
                # Reset optimizer state
                optimizer = self.optimizer([x_perturbation], lr=self.lr)

            x_perturbed = x + x_perturbation#self.surject_perturbation(x_perturbation)
            prediction_logits, prediction_feats = self.model(x_perturbed, return_features=True)
            
            pred_classes = prediction_logits.argsort(dim=-1, descending=True) # [B, C]
            attack_successful = (pred_classes[:, :K] == attack_targets).all(dim=-1) # [B]
            attack_energy = x_perturbation.flatten(1).norm(dim=-1) # [B]

            attack_improved = attack_successful & (attack_energy <= best_energy)

            best_perturbations[attack_improved] = x_perturbation[attack_improved]
            has_successful_attack[attack_improved] = True
            best_energy[attack_improved] = attack_energy[attack_improved]

            loss = self.loss(logits_pred=prediction_logits,
                             feats_pred=prediction_feats, 
                             feats_pred_0=prediction_feats_0,
                             attack_targets=attack_targets, 
                             model=self.model, **precomputed_state)
            
            loss = loss * topk_coefs

            loss = loss.sum()
            
            pbar.set_description(f"Loss: {loss.item():.3f}")
            
            loss = loss + x_perturbation.flatten(1).square().sum()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # If we were successfull let's start taking the norm down
            topk_coefs[attack_improved] *= 0.75

            # Project perturbation to be within image limits
            with torch.no_grad():
                x_perturbed = x + x_perturbation
                x_perturbed = x_perturbed.clamp_(min=0., max=1.)

                x_perturbation.data = x_perturbed - x
            
        x_perturbed_best = x + best_perturbations
        prediction_logits, prediction_feats = self.model(x_perturbed_best, return_features=True)

        if self.config.dump_plots:
            if os.path.isdir(self.config.plot_out):
                shutil.rmtree(self.config.plot_out)

            if has_successful_attack.any():
                def dump_random_map():
                    os.makedirs(self.config.plot_out, exist_ok=True)

                    # selected_idx = best_energy.argmin()
                    successful_idxs = has_successful_attack.nonzero()[:, 0]

                    if self.config.plot_idx == "find":
                        selected_idx = successful_idxs[torch.randperm(len(successful_idxs))[0]]
                        # selected_idx = best_energy.argmin()
                    else:
                        selected_idx = int(self.config.plot_idx)

                    print ("Selected idx", selected_idx)

                    top_classes = prediction_logits_0[selected_idx].argsort(dim=-1, descending=True)
                    attack_targets_selected = attack_targets[selected_idx]

                    def imgnet_names(idxs):
                        return [imgnet_idx_to_name[int(idx)].split(",")[0] for idx in idxs]
                                    
                    top_class_names = imgnet_names(top_classes)[:K]
                    attack_targets_selected_names = imgnet_names(attack_targets_selected)

                    def plot_attn_map(attn_map):
                        attn_map = attn_map[0].mean(dim=0)[1:] # [196] get class tokens
                        attn_map = attn_map.view(14, 14)
                        attn_map = F.interpolate(
                            attn_map[None, None],
                            x.shape[-2:],
                            mode="bilinear"
                        ).view(x.shape[-2:])

                        plt.imshow(attn_map.detach().cpu(), alpha=0.5)

                    plt.figure()
                    plt.imshow(x[selected_idx].permute(1,2,0).flip(dims=(-1,)).detach().cpu())
                    plt.axis("off")
                    plt.savefig(f"{self.config.plot_out}/clean_image.png", bbox_inches="tight", pad_inches=0)

                    plt.figure()
                    plt.imshow(x_perturbed_best[selected_idx].permute(1,2,0).flip(dims=(-1,)).detach().cpu())
                    plt.axis("off")
                    plt.savefig(f"{self.config.plot_out}/perturbed_image.png", bbox_inches="tight", pad_inches=0)

                    plt.figure()
                    plt.imshow(best_perturbations[selected_idx].mean(dim=0).abs().detach().cpu(), cmap="hot")
                    plt.colorbar()
                    plt.savefig(f"{self.config.plot_out}/perturbation.png", bbox_inches="tight")

                    if isinstance(self.model, MMPretrainVisualTransformerWrapper):
                        attn_maps_clean = self.model.get_attention_maps(x)[-1][selected_idx]
                        attn_maps_attacked = self.model.get_attention_maps(x_perturbed_best)[-1][selected_idx]

                        plt.figure()
                        plt.imshow(x[selected_idx].permute(1,2,0).flip(dims=(-1,)).detach().cpu())
                        plot_attn_map(attn_maps_clean)
                        plt.axis("off")
                        plt.savefig(f"{self.config.plot_out}/clean_map.png", bbox_inches="tight", pad_inches=0)

                        plt.figure()
                        plt.imshow(x[selected_idx].permute(1,2,0).flip(dims=(-1,)).detach().cpu())
                        plot_attn_map(attn_maps_attacked)
                        plt.axis("off")
                        plt.savefig(f"{self.config.plot_out}/attacked_map.png", bbox_inches="tight", pad_inches=0)

                    with open(f'{self.config.plot_out}/clean_classes_names.txt', 'w') as f:
                        f.write(", ".join(top_class_names))

                    with open(f'{self.config.plot_out}/attack_targets_names.txt', 'w') as f:
                        f.write(", ".join(attack_targets_selected_names))

                    with open(f'{self.config.plot_out}/clean_classes_names.txt', 'w') as f:
                        f.write(", ".join(top_class_names))

                    with open(f'{self.config.plot_out}/selected_idx.txt', 'w') as f:
                        if isinstance(selected_idx, torch.Tensor):
                            selected_idx = selected_idx.item()

                        f.write(str(selected_idx))

                    with open(f'{self.config.plot_out}/energy.txt', 'w') as f:
                        f.write(str(best_energy[selected_idx].item()))
                    
                    C = prediction_logits_0.shape[-1]
                    class_idxs = torch.arange(C) + 1
                    clean_probs = prediction_logits_0[selected_idx].detach().cpu().softmax(dim=-1)
                    attacked_probs = prediction_logits[selected_idx].detach().cpu().softmax(dim=-1)

                    def label_classes(bars):
                        adjusted_heights = {}
                        for i, cls_idx in enumerate(attack_targets_selected.tolist()):
                            bar = bars[cls_idx]
                            height = bar.get_height()
                            ann_x = bar.get_x() + bar.get_width()

                            rotation = 90
                            font_size = 10

                            max_neighboring_height = -1
                            for other_cls_idx in attack_targets_selected.tolist():
                                if abs(cls_idx - other_cls_idx) <= 40 and cls_idx != other_cls_idx:
                                    if other_cls_idx in adjusted_heights and adjusted_heights[other_cls_idx] > max_neighboring_height:
                                        max_neighboring_height = adjusted_heights[other_cls_idx]
                            
                            if max_neighboring_height > 0:
                                height = max_neighboring_height + 0.05

                            adjusted_heights[cls_idx] = height

                            plt.text(ann_x, height, f"[{i}]", rotation=rotation,
                                    ha='center', va='bottom', fontsize=font_size, color='red')#.get_bbox_patch().get_height()
                            

                    plt.figure()
                    bars_clean = plt.bar(class_idxs, clean_probs, width=4)
                    plt.ylim(0,1)
                    label_classes(bars_clean)
                    plt.savefig(f"{self.config.plot_out}/clean_probs.png", bbox_inches="tight", pad_inches=0)

                    plt.figure()
                    bars_attacked = plt.bar(class_idxs, attacked_probs, width=4)
                    plt.ylim(0,1)
                    label_classes(bars_attacked)
                    plt.savefig(f"{self.config.plot_out}/attacked_probs.png", bbox_inches="tight", pad_inches=0)

                    print ("Idx", selected_idx)
                    print (best_energy[selected_idx])
                    print ("Finished plotting")

                dump_random_map()
                import sys
                sys.exit(1)
                print ("Dumped attention map")


        return prediction_logits, best_perturbations, best_energy
    
    def forward(self, x, attack_targets, gt_labels):
        """
        This function is in charge of performing a binary search through
        topk loss coefficients and running attacks on each.
        """
        B = x.shape[0]
        device = x.device
        topk_coefs_lower = torch.full((B,), fill_value=self.topk_loss_coef_lower,
                                      device=device, dtype=torch.float)
        
        topk_coefs_upper = torch.full((B,), fill_value=self.topk_loss_coef_upper,
                                      device=device, dtype=torch.float)
        
        best_perturbations = torch.zeros_like(x) # [B, 3, H, W]
        best_energy = torch.full((B,), float('inf'), device=device) # [B]
        best_prediction_logits = None

        for search_step_i in range(self.binary_search_steps):
            if x.device.index is None or x.device.index == 0:
                print ("Running binary search step", search_step_i + 1)

            current_topk_coefs = (topk_coefs_lower + topk_coefs_upper) / 2
            current_logits, current_perturbations, current_energy = \
                self.attack(x, attack_targets, gt_labels, current_topk_coefs)
            
            current_attack_suceeded = ~torch.isinf(current_energy)
            
            update_mask = current_energy < best_energy

            best_perturbations[update_mask] = current_perturbations[update_mask]
            best_energy[update_mask] = current_energy[update_mask]
            
            if best_prediction_logits is None:
                best_prediction_logits = current_logits.clone()
            else:
                best_prediction_logits[update_mask] = current_logits[update_mask]

            # If we fail to attack, we must increase our topk coef
            topk_coefs_lower[~current_attack_suceeded] = current_topk_coefs[~current_attack_suceeded]

            # If we succeed, we must lower to seek a more frugal attack
            topk_coefs_upper[current_attack_suceeded] = current_topk_coefs[current_attack_suceeded]

            idist.barrier()
            
        return best_prediction_logits, best_perturbations