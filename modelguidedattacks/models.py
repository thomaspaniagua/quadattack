from torchvision import models

from modelguidedattacks.guides.instance_guide import InstanceGuide
from modelguidedattacks.guides.unguided import Unguided
from modelguidedattacks import losses

from .cls_models.registry import get_model

guide_model_registry = {
    "instance_guided": InstanceGuide,
    "unguided": Unguided
}

loss_registry = {
    "cvxproj": losses.CVXProjLoss,
    "cwk": losses.CWExtensionLoss,
    "ad": losses.AdversarialDistillationLoss
}

def setup_model(config, device):
    model = get_model(config.dataset, config.model, device)

    kwargs = {}

    if config.guide_model == "unguided":
        kwargs["iterations"] = config.unguided_iterations
        kwargs["lr"] = config.unguided_lr
        kwargs["loss_fn"] = loss_registry[config.loss]
        kwargs["binary_search_steps"] = config.binary_search_steps
        kwargs["topk_loss_coef_upper"] = config.topk_loss_coef_upper

    return guide_model_registry[config.guide_model](model, config, **kwargs)