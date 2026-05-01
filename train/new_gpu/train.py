# -*- coding: utf-8 -*-
import os
import click
import logging
from functools import partial
from contextlib import nullcontext

import wandb

import torch
import torch.nn as nn
import torch.optim as opt
import torch.distributed as dist

import kornia.augmentation as K

from data_preprocessing import FITSDataset, get_data_loader
from cnn import model_factory, model_stats, save_trained_model
from create_trainer import create_trainer, create_transfer_learner
from utils import specify_dropout_rate


def setup_ddp():
    """Initializes the distributed backend and sets the device."""
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "DDP launch variables are missing. "
            "Run with torchrun, e.g.:\n"
            "torchrun --standalone --nproc-per-node=4 train.py"
        )

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    return local_rank, device


@click.command()
@click.option("--experiment_name", type=str, default="demo")
@click.option("--run_id", type=str, default=None)
@click.option("--run_name", type=str, default=None)
@click.option("--model_type", type=click.Choice(["dragon"], case_sensitive=False), default="dragon")
@click.option("--model_state", type=click.Path(exists=True), default=None)
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option("--split_slug", type=str, required=True)
@click.option("--cutout_size", type=int, default=94)
@click.option("--channels", type=int, default=1)
@click.option("--n_classes", type=int, default=6)
@click.option("--n_workers", type=int, default=4)
@click.option("--loss", type=click.Choice(["nll", "ce"], case_sensitive=False), default="ce")
@click.option("--batch_size", type=int, default=16)
@click.option("--epochs", type=int, default=40)
@click.option("--lr", type=float, default=5e-7)
@click.option("--momentum", type=float, default=0.9)
@click.option("--weight_decay", type=float, default=0)
@click.option("--parallel/--no-parallel", default=True)
@click.option("--normalize/--no-normalize", default=True)
@click.option("--crop/--no-crop", default=True)
@click.option("--nesterov/--no-nesterov", default=False)
@click.option("--dropout_rate", type=float, default=None)
@click.option("--force_reload/--no_force_reload", default=False)
@click.option("--expand_data", type=int, default=1)
@click.option("--train/--transfer_learn", default=True)
@click.option("--scheduler/--no_scheduler", default=True)
def train(**kwargs):
    """Runs the training procedure using DDP."""

    # 1. Setup DDP Environment
    local_rank, device = setup_ddp()
    is_main_process = local_rank == 0

    # Only log setup info on the main process
    if is_main_process:
        log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        logging.basicConfig(level=logging.INFO, format=log_fmt)
        logging.info("Distributed Data Parallel Initialized.")

    args = {k: v for k, v in kwargs.items()}
    args["device"] = device

    # Create the model given model_type
    cls = model_factory(args["model_type"])
    model_args = {
        "cutout_size": args["cutout_size"],
        "channels": args["channels"],
        "num_classes": args["n_classes"]
    }

    if "drp" in args["model_type"].split("_"):
        if is_main_process:
            logging.info(f"Using dropout rate of {args['dropout_rate']} in the model")
        model_args["dropout"] = "True"

    model = cls(**model_args)
    model = model.to(device)

    if args["dropout_rate"] is not None:
        specify_dropout_rate(model, args["dropout_rate"])

    # Load the model from a saved state if provided (map to local device)
    if args["model_state"]:
        if is_main_process:
            logging.info(f'Loading model from {args["model_state"]}...')
        model.load_state_dict(torch.load(args["model_state"], map_location=device))

    # Wrap model in DDP
    model = nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank],
        output_device=local_rank
    )

    # Define the optimizer
    optimizer = opt.SGD(
        model.parameters(),
        lr=args["lr"],
        momentum=args["momentum"],
        nesterov=args["nesterov"],
        weight_decay=args["weight_decay"],
    )

    # Select the desired transforms
    T = None
    if args["crop"]:
        T = [K.CenterCrop(args["cutout_size"]),
             K.RandomHorizontalFlip(),
             K.RandomVerticalFlip(),
             K.RandomRotation(360)]

    # Generate the Datasets
    splits = ("train", "devel", "test")
    datasets = {
        k: FITSDataset(
            data_dir=args["data_dir"],
            slug=args["split_slug"],
            cutout_size=args["cutout_size"],
            channels=args["channels"],
            normalize=args["normalize"],
            transforms=T,
            split=k,
            num_classes=args["n_classes"],
            expand_factor=args["expand_data"] if k == "train" else 1,
            force_reload=args["force_reload"]
        )
        for k in splits
    }

    # Create a DataLoader factory based on command-line args
    loader_factory = partial(
        get_data_loader,
        batch_size=args["batch_size"],
        n_workers=args["n_workers"],
    )

    # Generate Loaders (Crucial: disable shuffle if sampler is used)
    loaders = {}
    for k, v in datasets.items():
        sampler = v.get_sampler()
        # DataLoader requires shuffle=False if a sampler is provided
        do_shuffle = True if (k == 'train' and sampler is None) else False
        loaders[k] = loader_factory(v, shuffle=do_shuffle, sampler=sampler)

    args["splits"] = {k: len(v.dataset) for k, v in loaders.items()}

    # Define the criterion
    loss_dict = {
        "nll": nn.NLLLoss(),
        "ce": nn.CrossEntropyLoss(),
    }
    criterion = loss_dict[args["loss"]]

    # Setup W&B Context (Only initialize for rank 0)
    if is_main_process:
        wandb.login()
        run_ctx = wandb.init(
            project=args["experiment_name"],
            id=args["run_id"],
            resume="allow",
            config={
                "num_classes": args["n_classes"],
                "architecture": "CNN",
                "parameters": {
                    "learning_rate": args["lr"],
                    "momentum": args["momentum"],
                    "nesterov": args["nesterov"],
                    "weight_decay": args["weight_decay"],
                    "epochs": args["epochs"],
                    "batch_size": args["batch_size"]
                }
            }
        )
    else:
        run_ctx = nullcontext()

    # Enter Training Block
    with run_ctx as run:
        if is_main_process:
            # model.module is required to unwrap the DDP wrapper for stats
            args = {**args, **model_stats(model.module)}
            run.log(args)

        # Set up trainer
        if args["train"]:
            if is_main_process: logging.info("Creating trainer...")
            trainer = create_trainer(
                model, optimizer, criterion, loaders, args["device"], args["scheduler"]
            )
        else:
            if is_main_process: logging.info("Creating trainer and freezing layers for transfer learning...")
            trainer = create_transfer_learner(
                model, optimizer, criterion, loaders, args["device"], args["scheduler"]
            )

        # Run trainer
        trainer.run(loaders["train"], max_epochs=args["epochs"])

        # Save model state only on main process
        if is_main_process:
            slug = f"{args['experiment_name']}-{args['split_slug']}-{run.id}"

            # Save the unwrapped model
            model_path = save_trained_model(model.module, slug)

            logging.info(f"Saved model to {model_path}")
            run.log_artifact(model_path)
            wandb.finish()

    # Cleanup DDP
    dist.destroy_process_group()


if __name__ == "__main__":
    train()