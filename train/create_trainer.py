import wandb

from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Loss, Accuracy, Precision, ConfusionMatrix, Recall

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
        "cm": ConfusionMatrix(num_classes=wandb.config["num_classes"], output_transform=lambda x: x)
    }

    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )

    # Function to log metrics to wandb
    def log_metrics(trainer, loader, log_prefix=""):
        evaluator.run(loader)
        metrics = evaluator.state.metrics
        log_dict = {k: v for k, v in metrics.items() if k != "cm"}

        # Handle the confusion matrix separately
        cm = metrics["cm"].cpu().numpy()
        class_names = [str(i) for i in range(wandb.config["num_classes"])]
        cm_plot = wandb.plot.confusion_matrix(probs=None,
                                              y_true=cm.argmax(axis=1),
                                              preds=cm.argmax(axis=0),
                                              class_names=class_names)

        # Log other metrics and the confusion matrix plot
        log_dict[f"{log_prefix}confusion_matrix"] = cm_plot
        wandb.log(log_dict)

    # Define training hooks
    @trainer.on(Events.STARTED)
    def log_results_start(trainer):
        for L, loader in loaders.items():
            log_metrics(trainer, loader, log_prefix=f"{L}_")

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_devel_results(trainer):
        log_metrics(trainer, loaders["devel"], log_prefix="devel_")

    @trainer.on(Events.COMPLETED)
    def log_results_end(trainer):
        for L, loader in loaders.items():
            log_metrics(trainer, loader, log_prefix=f"{L}_")

    return trainer
