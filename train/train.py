# -*- coding: utf-8 -*-
import click
import logging
import math
from pathlib import Path
from functools import partial

import wandb

import torch
import torch.nn as nn
import torch.optim as opt

import kornia.augmentation as K

from data_preprocessing import FITSDataset, get_data_loader
from cnn import model_factory, model_stats, save_trained_model
from create_trainer import create_trainer, create_transfer_learner
from utils import discover_devices, specify_dropout_rate


import random
import numpy as np

@click.command()
@click.option("--experiment_name", type=str, default="demo")
@click.option(
    "--run_id",
    type=str,
    default=None,
    help="""The run id. Practically this only needs to be used
if you are resuming a previosuly run experiment""",
)
@click.option(
    "--run_name",
    type=str,
    default=None,
    help="""A run is supposed to be a sub-class of an experiment.
So this variable should be specified accordingly""",
)
@click.option(
    "--model_type",
    type=click.Choice(
        [
            "dragon"
        ],
        case_sensitive=False,
    ),
    default="dragon",
)
@click.option("--model_state", type=click.Path(exists=True), default=None)
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option(
    "--split_slug",
    type=str,
    required=True,
    help="""This specifies how the data_preprocessing is split into train/
devel/test sets. Balanced/Unbalanced refer to whether selecting
equal number of images from each class. xs, sm, lg, dev all refer
to what fraction is picked for train/devel/test.""",
)
@click.option("--cutout_size", type=int, default=94)
@click.option("--channels", type=int, default=1)
@click.option("--n_classes", type=int, default=6)
@click.option(
    "--n_workers",
    type=int,
    default=4,
    help="""The number of workers to be used during the
data_preprocessing loading process.""",
)
@click.option(
    "--loss",
    type=click.Choice(
        [
            "nll",
            "ce",
        ],
        case_sensitive=False,
    ),
    default="ce",
    help="""The loss function to use""",
)
@click.option("--batch_size", type=int, default=16)
@click.option("--epochs", type=int, default=40)
@click.option("--lr", type=float, default=5e-7)
@click.option("--momentum", type=float, default=0.9)
@click.option("--weight_decay", type=float, default=0)
@click.option(
    "--parallel/--no-parallel",
    default=True,
    help="""The parallel argument controls whether or not
to use multiple GPUs when they are available""",
)
@click.option(
    "--normalize/--no-normalize",
    default=True,
    help="""The normalize argument controls whether or not, the
loaded images will be normalized using the arsinh function""",
)
@click.option(
    "--crop/--no-crop",
    default=True,
    help="""If True, all images are passed through a cropping
operation before being fed into the network. Images are cropped
to the cutout_size parameter""",
)
@click.option(
    "--nesterov/--no-nesterov",
    default=False,
    help="""Whether to use Nesterov momentum or not""",
)
@click.option(
    "--dropout_rate",
    type=float,
    default=None,
    help="""The dropout rate to use for all the layers in the
    model. If this is set to None, then the default dropout rate
    in the specific model is used.""",
)
@click.option(
    "--force_reload/--no_force_reload",
    default=False,
)
@click.option(
    "--expand_data",
    type=int,
    default=1,
    help="""This controls the factor by which the training
data is augmented""",
)
@click.option(
    "--train/--transfer_learn",
    default=True,
    help="""Specifies whether you wish to do transfer learning. If transfer learning,
    you must specify model path in the model_state argument."""
)
@click.option(
    "--scheduler/--no_scheduler",
    default=True
)
def train(**kwargs):
    """Runs the training procedure using MLFlow."""

    # Copy and log args
    args = {k: v for k, v in kwargs.items()}

    # Discover devices
    args["device"] = discover_devices()

    # Create the model given model_type
    cls = model_factory(args["model_type"])
    model_args = {
        "cutout_size": args["cutout_size"],
        "channels": args["channels"],
        "num_classes": args["n_classes"]
    }

    if "drp" in args["model_type"].split("_"):
        logging.info(
            "Using dropout rate of {} in the model".format(
                args["dropout_rate"]
            )
        )
        model_args["dropout"] = "True"

    model = cls(**model_args)
    model = nn.DataParallel(model) if args["parallel"] else model
    model = model.to(args["device"])

    # Chnaging the default dropout rate if specified
    if args["dropout_rate"] is not None:
        specify_dropout_rate(model, args["dropout_rate"])

    # Load the model from a saved state if provided
    if args["model_state"]:
        logging.info(f'Loading model from {args["model_state"]}...')
        if args["device"] == "cpu":
            model.load_state_dict(torch.load(args["model_state"], map_location="cpu"))
        else:
            model.load_state_dict(torch.load(args["model_state"]))

    # Define the optimizer
    optimizer = opt.SGD(
        model.parameters(),
        lr=args["lr"],
        momentum=args["momentum"],
        nesterov=args["nesterov"],
        weight_decay=args["weight_decay"],
    )

    # Create a DataLoader factory based on command-line args
    loader_factory = partial(
        get_data_loader,
        batch_size=args["batch_size"],
        n_workers=args["n_workers"],
    )

    # Select the desired transforms
    T = None
    if args["crop"]:
        T = [K.CenterCrop(args["cutout_size"]),
            K.RandomHorizontalFlip(),
            K.RandomVerticalFlip(),
            K.RandomRotation(360)]

    # Generate the DataLoaders and log the train/devel/test split sizes
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
    loaders = {k: loader_factory(v, shuffle=(k == 'train')) for k, v in datasets.items()}
    args["splits"] = {k: len(v.dataset) for k, v in loaders.items()}

    # Define the criterion
    loss_dict = {
        "nll": nn.NLLLoss(),
        "ce": nn.CrossEntropyLoss(),
    }
    criterion = loss_dict[args["loss"]]

    # Log into W&B
    wandb.login()

    # Initializing W&B run
    with wandb.init(
        project=args["experiment_name"],
        id=args["run_id"],
        resume="allow",

        # track hyperparameters and run metadata
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
    ) as run:
        # Write the parameters and model stats to W&B
        args = {**args, **model_stats(model)}
        run.log(args)

        # Set up trainer
        if args["train"]:
            logging.info("Creating trainer...")
            trainer = create_trainer(
                model, optimizer, criterion, loaders, args["device"], args["scheduler"]
            )
        else:
            logging.info("Creating trainer and freezing layers for transfer learning...")
            trainer = create_transfer_learner(
                model, optimizer, criterion, loaders, args["device"], args["scheduler"]
            )

        # Run trainer and save model state
        trainer.run(loaders["train"], max_epochs=args["epochs"])
        slug = (
            f"{args['experiment_name']}-{args['split_slug']}-"
            f"{run.id}"
        )

        model_path = save_trained_model(model, slug)

        # Log model as an artifact
        logging.info(f"Saved model to {model_path}")
        run.log_artifact(model_path)

        # Finish the W&B run!
        wandb.finish()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    train()
