import wandb
import os
import torch
import torch.distributed as dist

from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Loss, Accuracy, Precision, ConfusionMatrix, Recall, Fbeta
from ignite.contrib.handlers import ProgressBar
from ignite.contrib.handlers.param_scheduler import LRScheduler
import logging

from torch.optim.lr_scheduler import CosineAnnealingLR


def is_main_process():
    if not dist.is_available() or not dist.is_initialized():
        return True
    return dist.get_rank() == 0


def create_trainer(model, optimizer, criterion, loaders, device, use_scheduler=True):
    """Set up Ignite trainer and evaluator."""
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device
    )

    # Only attach progress bars on the main process to prevent console spam
    if is_main_process():
        pbar = ProgressBar(persist=False)
        pbar.attach(trainer, output_transform=lambda x: {'batch_loss': x})

    if use_scheduler:
        torch_lr_scheduler = CosineAnnealingLR(optimizer, T_max=20)
        scheduler = LRScheduler(torch_lr_scheduler)

    metrics = {
        "accuracy": Accuracy(),
        "precision": Precision(average="weighted"),
        "recall": Recall(average="weighted"),
        "loss": Loss(criterion),
        # Use wandb.config only if initialized
        "cm": ConfusionMatrix(num_classes=wandb.config["num_classes"] if is_main_process() else 6,
                              output_transform=lambda x: x),
        "f1": Fbeta(beta=1)
    }

    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )

    if is_main_process():
        pbar_eval = ProgressBar(persist=False, desc="Evaluating")
        pbar_eval.attach(evaluator)

    def log_metrics(trainer, loader, log_prefix=""):
        # Evaluate on all processes (required for DDP sync)
        evaluator.state.metrics = {}
        evaluator.run(loader)

        # Only log to WandB on the main process
        if is_main_process():
            logging.info(f"Logging metrics for {log_prefix}")
            metrics = evaluator.state.metrics
            log_dict = {f"{log_prefix}{k}": v for k, v in metrics.items() if k != "cm"}

            cm = metrics["cm"].cpu().numpy()
            class_names = [str(i) for i in range(wandb.config["num_classes"])]

            y_true, y_pred = [], []
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    y_true.extend([i] * int(cm[i, j]))
                    y_pred.extend([j] * int(cm[i, j]))

            cm_plot = wandb.plot.confusion_matrix(probs=None,
                                                  y_true=y_true,
                                                  preds=y_pred,
                                                  class_names=class_names)

            log_dict[f"{log_prefix}confusion_matrix"] = cm_plot
            wandb.log(log_dict)

    def get_current_lr(optimizer):
        return optimizer.param_groups[0]['lr']

    # --- DDP CRITICAL: Sync Sampler Epochs ---
    @trainer.on(Events.EPOCH_STARTED)
    def set_epoch(engine):
        # DistributedSampler needs to know the epoch to shuffle data properly
        for loader in loaders.values():
            if hasattr(loader, 'sampler') and hasattr(loader.sampler, 'set_epoch'):
                loader.sampler.set_epoch(engine.state.epoch)

    if use_scheduler:
        trainer.add_event_handler(Events.ITERATION_STARTED, scheduler)

    @trainer.on(Events.STARTED)
    def log_results_start(trainer):
        if is_main_process(): logging.info("Log results started.")
        for L, loader in loaders.items():
            log_metrics(trainer, loader, log_prefix=f"{L}_")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_devel_results(trainer):
        for L, loader in loaders.items():
            log_metrics(trainer, loader, log_prefix=f"{L}_")
        if is_main_process():
            wandb.log({"lr": get_current_lr(optimizer)})

    @trainer.on(Events.ITERATION_COMPLETED)
    def clip_gradients(engine):
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)

    @trainer.on(Events.COMPLETED)
    def log_results_end(trainer):
        for L, loader in loaders.items():
            log_metrics(trainer, loader, log_prefix=f"{L}_")

        if is_main_process(): logging.info("Terminating run explicitly.")
        trainer.terminate()

    return trainer


def create_transfer_learner(model, optimizer, criterion, loaders, device):
    """Method to create a transfer learner trainer."""
    frozen_layer_stack = []

    if is_main_process(): logging.info("Freezing non-FC layers for given model...")

    # model.module handles the underlying model wrapped inside DDP
    target_model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    for name, param in target_model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
            frozen_layer_stack.append((name, param))

    trainer = create_trainer(
        model, optimizer, criterion, loaders, device
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def unfreeze_layers(engine):
        epoch = engine.state.epoch

        if is_main_process(): wandb.log({"frozen_layers": len(frozen_layer_stack)})

        if frozen_layer_stack:
            top_name, top_param = frozen_layer_stack[-1]
            layer_name = top_name.split('.')[1]
            while frozen_layer_stack and frozen_layer_stack[-1][0].split('.')[1] == layer_name:
                name, param = frozen_layer_stack.pop()
                param.requires_grad = True
                if is_main_process(): logging.info(f"Epoch[{epoch}]: layer {name} is now trainable.")
        else:
            if is_main_process(): logging.info(f"Epoch[{epoch}]: all layers trainable.")

    return trainer