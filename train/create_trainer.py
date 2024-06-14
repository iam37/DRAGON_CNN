import wandb

from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Loss, Accuracy, Precision, ConfusionMatrix, Recall, Fbeta
from ignite.handlers import TerminateOnNan
import torch.nn as nn
import logging

def create_trainer(model, optimizer, criterion, loaders, device):
    """Set up Ignite trainer and evaluator."""
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device
    )

    metrics = {
        "accuracy": Accuracy(),
        "precision": Precision(average="weighted"),
        "recall": Recall(average="weighted"),
        "loss": Loss(criterion),
        "cm": ConfusionMatrix(num_classes=wandb.config["num_classes"], output_transform=lambda x: x),
        "f1": Fbeta(beta=1)
    }

    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )

    # Function to log metrics to wandb
    def log_metrics(trainer, loader, log_prefix=""):
        logging.info(f"Logging metrics for {log_prefix}")

        # Reset evaluator state before running evaluation
        evaluator.state.metrics = {}

        evaluator.run(loader)
        metrics = evaluator.state.metrics
        log_dict = {f"{log_prefix}{k}": v for k, v in metrics.items() if k != "cm"}

        # Handle the confusion matrix separately
        cm = metrics["cm"].cpu().numpy()
        class_names = [str(i) for i in range(wandb.config["num_classes"])]

        # Calculate true and predicted labels from the confusion matrix
        y_true, y_pred = [], []
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                y_true.extend([i] * int(cm[i, j]))
                y_pred.extend([j] * int(cm[i, j]))

        cm_plot = wandb.plot.confusion_matrix(probs=None,
                                              y_true=y_true,
                                              preds=y_pred,
                                              class_names=class_names)

        # Log other metrics and the confusion matrix plot
        log_dict[f"{log_prefix}confusion_matrix"] = cm_plot
        wandb.log(log_dict)

    # Define training hooks
    @trainer.on(Events.STARTED)
    def log_results_start(trainer):
        logging.info("Log results started.")
        for L, loader in loaders.items():
            log_metrics(trainer, loader, log_prefix=f"{L}_")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_devel_results(trainer):
        for L, loader in loaders.items():
            log_metrics(trainer, loader, log_prefix=f"{L}_")

    @trainer.on(Events.ITERATION_COMPLETED)
    def clip_gradients(engine):
        for param in model.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)

    @trainer.on(Events.COMPLETED)
    def log_results_end(trainer):
        for L, loader in loaders.items():
            log_metrics(trainer, loader, log_prefix=f"{L}_")

        logging.info("Terminating run explicitly.")
        trainer.terminate()

    return trainer


def create_transfer_learner(model, optimizer, criterion, loaders, device):
    """Method to create a transfer learner trainer."""

    # Initialize a stack that contains all frozen layers.
    frozen_layer_stack = []

    # Initial freezing of the layers.
    logging.info("Freezing non-FC layers for given model...")
    for name, param in model.named_parameters():
        if "fc" not in name:
            param.requires_grad = False
            frozen_layer_stack.append((name, param))

    # Create trainer.
    trainer = create_trainer(
        model, optimizer, criterion, loaders, device
    )

    # Gradual unfreezing of layers based on epoch.
    @trainer.on(Events.EPOCH_COMPLETED)
    def unfreeze_layers(engine):
        epoch = engine.state.epoch
        if frozen_layer_stack:
            name, param = frozen_layer_stack.pop()
            param.requires_grad = True
            logging.info(f"Epoch {epoch}: layer {name} is now trainable.")
        else:
            logging.info(f"Epoch {epoch}: all layers trainable.")

    return trainer
