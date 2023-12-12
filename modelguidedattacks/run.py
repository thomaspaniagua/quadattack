import sys
from pprint import pformat
from typing import Any
import os
import torch

import ignite.distributed as idist
import yaml
from ignite.engine import Events
from ignite.metrics import Accuracy, Loss
from ignite.utils import manual_seed
from torch import nn, optim

from modelguidedattacks.data.setup import setup_data
from modelguidedattacks.losses.boilerplate import BoilerplateLoss
from modelguidedattacks.losses.energy import Energy, EnergyLoss
from modelguidedattacks.metrics.topk_accuracy import TopKAccuracy
from modelguidedattacks.models import setup_model
from modelguidedattacks.trainers import setup_evaluator, setup_trainer
from modelguidedattacks.utils import setup_parser, setup_output_dir
from modelguidedattacks.utils import setup_logging, log_metrics, Engine

def run(local_rank: int, config: Any):

    print ("Running ", local_rank)
    # make a certain seed
    rank = idist.get_rank()
    manual_seed(config.seed + rank)

    # create output folder
    config.output_dir = setup_output_dir(config, rank)

    # setup engines logger with python logging
    # print training configurations
    logger = setup_logging(config)
    logger.info("Configuration: \n%s", pformat(vars(config)))
    (config.output_dir / "config-lock.yaml").write_text(yaml.dump(config))

    # donwload datasets and create dataloaders
    dataloader_train, dataloader_eval = setup_data(config, rank)

    # model, optimizer, loss function, device
    device = idist.device()
    model = idist.auto_model(setup_model(config, idist.device()))
    loss_fn = BoilerplateLoss().to(device=device)
    l2_energy_loss = Energy(p=2).to(device)
    l1_energy_loss = Energy(p=1).to(device)
    l_inf_energy_loss = Energy(p=torch.inf).to(device)

    evaluator = setup_evaluator(config, model, device)
    evaluator.logger = logger

    # attach metrics to evaluator
    accuracy = TopKAccuracy(device=device)
    metrics = {
        "ASR": accuracy,
        "L2 Energy": EnergyLoss(l2_energy_loss, device=device),
        "L1 Energy": EnergyLoss(l1_energy_loss, device=device),
        "L_inf Energy": EnergyLoss(l_inf_energy_loss, device=device),

        "L2 Energy Min": EnergyLoss(l2_energy_loss, reduction="min", device=device),
        "L1 Energy Min": EnergyLoss(l1_energy_loss, reduction="min", device=device),
        "L_inf Energy Min": EnergyLoss(l_inf_energy_loss, reduction="min", device=device),

        "L2 Energy Max": EnergyLoss(l2_energy_loss, reduction="max", device=device),
        "L1 Energy Max": EnergyLoss(l1_energy_loss, reduction="max", device=device),
        "L_inf Energy Max": EnergyLoss(l_inf_energy_loss, reduction="max", device=device)
    }
    for name, metric in metrics.items():
        metric.attach(evaluator, name)

    if config.guide_model in ["unguided", "instance_guided"]:

        first_batch_passed = False
        early_stopped = False

        def compute_metrics(engine: Engine, tag: str):
            nonlocal first_batch_passed
            nonlocal early_stopped

            for name, metric in metrics.items():
                metric.completed(engine, name)

            if not first_batch_passed:
                if engine.state.metrics["ASR"] < 1e-3:
                    print ("Early stop, assuming no success throughout")
                    early_stopped = True
                    engine.terminate()
                else:
                    first_batch_passed = True

        evaluator.add_event_handler(
            Events.ITERATION_COMPLETED(every=config.log_every_iters),
            compute_metrics,
            tag="eval",
        )

        evaluator.add_event_handler(
            Events.ITERATION_COMPLETED(every=config.log_every_iters),
            log_metrics,
            tag="eval",
        )

        evaluator.run(dataloader_eval, epoch_length=config.eval_epoch_length)
        log_metrics(evaluator, "eval")

        if len(config.out_dir) > 0:
            # Store results in out_dir
            os.makedirs(config.out_dir, exist_ok=True)
            metrics_dict = evaluator.state.metrics
            metrics_dict["config"] = config
            metrics_dict["early_stopped"] = early_stopped

            metrics_file_path = os.path.join(config.out_dir, "results.save")
            torch.save(metrics_dict, metrics_file_path)

        # No need to train with an unguided model
        return

    assert False, "This code path is for the future"

# main entrypoint
def launch(config=None):
    if config is None:
        config_path = sys.argv[1]
        config = setup_parser(config_path).parse_args(sys.argv[2:])
        
    backend = config.backend
    nproc_per_node = config.nproc_per_node

    if nproc_per_node == 0 or backend is None:
        backend = None
        nproc_per_node = None

    with idist.Parallel(backend, nproc_per_node) as p:
        p.run(run, config=config)


if __name__ == "__main__":
    launch()
