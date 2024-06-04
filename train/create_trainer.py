import wandb

from ignite.engine import (
    Events,
    create_supervised_trainer,
    create_supervised_evaluator,
)
from ignite.metrics import Loss, Accuracy, Precision

def create_trainer(model, optimizer, criterion, loaders, device):
    """Set up Ignite trainer and evaluator."""
    trainer = create_supervised_trainer(
        model, optimizer, criterion, device=device
    )

    metrics = {
        "accuracy": Accuracy(),
        "precision": Precision(average=False),
        "loss": Loss(criterion)
    }

    evaluator = create_supervised_evaluator(
        model, metrics=metrics, device=device
    )

    # Define training hooks
    @trainer.on(Events.STARTED)
    def log_results_start(trainer):
        for L, loader in loaders.items():
            evaluator.run(loader)
            metrics = evaluator.state.metrics
            log_dict = {k: v for k, v in zip(metrics.keys(), metrics.values())}
            wandb.log(log_dict)

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_devel_results(trainer):
        evaluator.run(loaders["devel"])
        metrics = evaluator.state.metrics
        log_dict = {k: v for k, v in zip(metrics.keys(), metrics.values())}
        wandb.log(log_dict)

    @trainer.on(Events.COMPLETED)
    def log_results_end(trainer):
        for L, loader in loaders.items():
            evaluator.run(loader)
            metrics = evaluator.state.metrics
            log_dict = {k: v for k, v in zip(metrics.keys(), metrics.values())}
            wandb.log(log_dict)

        wandb.finish()

    return trainer
