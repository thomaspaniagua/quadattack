from typing import Any, Union

import ignite.distributed as idist
import torch
from ignite.engine import DeterministicEngine, Engine, Events
from torch.cuda.amp import autocast
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DistributedSampler, Sampler


def setup_trainer(
    config: Any,
    model: Module,
    optimizer: Optimizer,
    loss_fn: Module,
    device: Union[str, torch.device],
    train_sampler: Sampler,
) -> Union[Engine, DeterministicEngine]:
    def train_function(engine: Union[Engine, DeterministicEngine], batch: Any):
        if config.overfit:
            # No batch norm
            model.eval()
        else:
            model.train()

        samples = batch[0].to(device, non_blocking=True)
        targets = batch[1].to(device, non_blocking=True)
        attack_targets = batch[2].to(device, non_blocking=True)
        sample_ids = batch[3].to(device, non_blocking=True)

        with autocast(config.use_amp):
            outputs = model(samples, attack_targets)
            loss = loss_fn(outputs, attack_targets, targets)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss = loss.item()
        engine.state.metrics = {
            "epoch": engine.state.epoch,
            "train_loss": train_loss,
        }
        return {"train_loss": train_loss}


    trainer = Engine(train_function)

    # set epoch for distributed sa5mpler
    @trainer.on(Events.EPOCH_STARTED)
    def set_epoch():
        if idist.get_world_size() > 1 and isinstance(train_sampler, DistributedSampler):
            train_sampler.set_epoch(trainer.state.epoch - 1)

    return trainer


def setup_evaluator(
    config: Any,
    model: Module,
    device: Union[str, torch.device],
) -> Engine:
    @torch.no_grad()
    def eval_function(engine: Engine, batch: Any):
        model.eval()

        samples, gt_labels, attack_targets, sample_ids = batch

        samples = samples.to(device, non_blocking=True)
        gt_labels = gt_labels.to(device, non_blocking=True)
        attack_targets = attack_targets.to(device, non_blocking=True)
        sample_ids = sample_ids.to(device, non_blocking=True)

        with autocast(config.use_amp):
            outputs, perturbations = model(samples, attack_targets, gt_labels)

        return outputs, attack_targets, {
            "gt_targets": gt_labels,
            "perturbations": perturbations
        }

    return Engine(eval_function)
