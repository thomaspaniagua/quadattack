import mmpretrain
import torch
from torch import nn
from collections.abc import Iterable
from mmpretrain.models.utils.attention import MultiheadAttention
# This holds model instantiation functions by (dataset_name, model_name) tuple keys
MODEL_REGISTRY = {}

class ClsModel(nn.Module):
    dataset_name: str
    model_name: str

    def __init__(self, dataset_name: str, model_name: str, device: str) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.model_name = model_name
        self.device = device

    def head_features(self):
        pass

    def num_classes(self):
        pass

    def forward(self, x):
        """
        x: [B, 3 (RGB), H, W] image (float) [0,1]

        returns: [B, C] class logits
        """

        raise NotImplementedError("Forward not implemented for base class")

class MMPretrainModelWrapper(ClsModel):
    """
    Calls data preprocessing for model before entering forward
    """
    def __init__(self, model: nn.Module, dataset_name: str, model_name: str, device: str) -> None:
        super().__init__(dataset_name, model_name, device)
        self.model = model

    @property
    def final_linear_layer(self):
        return self.model.head.fc

    def head_features(self):
        return self.final_linear_layer.in_features
    
    def num_classes(self):
        return self.final_linear_layer.out_features
    
    def head(self, feats):
        return self.model.head((feats,))
    
    def head_matrices(self):
        return self.final_linear_layer.weight, self.final_linear_layer.bias

    def forward(self, x, return_features=False):
        # Data preprocessor expects 0-255 range, but we don't want to cast to proper
        # uint8 because we want to maintain differentiability
        x = x * 255.
        x = self.model.data_preprocessor({"inputs": x})["inputs"]

        if return_features:
            feats = self.model.extract_feat(x)
            
            preds = self.model.head(feats)
            if isinstance(feats, Iterable):
                feats = feats[-1]
                
            return preds, feats
        else:
            return self.model(x)
        
class MMPretrainVisualTransformerWrapper(MMPretrainModelWrapper):
    def __init__(self, model, dataset_name: str, model_name: str, device: str) -> None:
        super().__init__(model, dataset_name, model_name, device)

        attn_layers = []

        def find_mha(m: nn.Module):
            if isinstance(m, MultiheadAttention):
                attn_layers.append(m)

        model.apply(find_mha)

        self.attn_layers = attn_layers

    @property
    def final_linear_layer(self):
        return self.model.head.layers.head
    
    def get_attention_maps(self, x):
        clean_forwards = []

        attention_maps = []

        for attn_layer in self.attn_layers:
            clean_forward = attn_layer.forward
            clean_forwards.append(clean_forward)

            def scaled_dot_prod_attn(query,
                                        key,
                                        value,
                                        attn_mask=None,
                                        dropout_p=0.,
                                        scale=None,
                                        is_causal=False):
                scale = scale or query.size(-1)**0.5
                if is_causal and attn_mask is not None:
                    attn_mask = torch.ones(
                        query.size(-2), key.size(-2), dtype=torch.bool).tril(diagonal=0)
                if attn_mask is not None and attn_mask.dtype == torch.bool:
                    attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf'))

                attn_weight = query @ key.transpose(-2, -1) / scale
                if attn_mask is not None:
                    attn_weight += attn_mask
                attn_weight = torch.softmax(attn_weight, dim=-1)

                attention_maps.append(attn_weight)

                attn_weight = torch.dropout(attn_weight, dropout_p, True)
                return attn_weight @ value

            attn_layer.scaled_dot_product_attention = scaled_dot_prod_attn

        ret_val = super().forward(x, False)

        for attn_layer, clean_forward in zip(self.attn_layers, clean_forwards):
            attn_layer.forward = clean_forward

        return attention_maps
    
def register_mmcls_model(config_name, dataset_name, model_name, 
                         wrapper_class=MMPretrainModelWrapper):
    def instantiate_model(device):
        model = mmpretrain.get_model(config_name, pretrained=True, device=device)
        wrapper = wrapper_class(model, dataset_name, model_name, device)
        return wrapper
    
    MODEL_REGISTRY[(dataset_name, model_name)] = instantiate_model

def register_default_models():
    register_mmcls_model("resnet18_8xb16_cifar10", "cifar10", "resnet18")
    register_mmcls_model("resnet34_8xb16_cifar10", "cifar10", "resnet34")
    register_mmcls_model("resnet18_8xb32_in1k", "imagenet", "resnet18")
    register_mmcls_model("resnet50_8xb16_cifar100", "cifar100", "resnet50")
    register_mmcls_model("resnet50_8xb32_in1k", "imagenet", "resnet50")
    register_mmcls_model("densenet121_3rdparty_in1k", "imagenet", "densenet121")

    register_mmcls_model("deit-small_4xb256_in1k", "imagenet", "deit_small",
                          wrapper_class=MMPretrainVisualTransformerWrapper)
    
    register_mmcls_model("vit-base-p16_32xb128-mae_in1k", "imagenet", "vit_base",
                          wrapper_class=MMPretrainVisualTransformerWrapper)

def get_model(dataset_name, model_name, device):
    """
    Returns instance of model pretrained with specified dataset
    """

    return MODEL_REGISTRY[(dataset_name, model_name)](device).eval()