# -*- coding: utf-8 -*-
import click
import logging
import math
from pathlib import Path
from functools import partial

import wandb
import os
import subprocess

import torch
import torch.nn as nn
import torch.optim as opt

import kornia.augmentation as K
import torch.multiprocessing as mp

from data_preprocessing import FITSDataset, get_data_loader
from cnn import model_factory, model_stats, save_trained_model
from train import create_trainer
from utils import discover_devices, specify_dropout_rate

# Global Sweep Configuration. This also effects early stopping
# for bad runs!
sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "accuracy"},
    "parameters": {
        "learning_rate": {"values": [0.001, 0.0001, 0.0005]},
        "momentum": {"values": [1e-4, 1e-5, 1e-6]},
        "nesterov": {"values": [True, False]},
        "weight_decay": {"values": [1e-6, 1e-8, 1e-10]},
        "epochs": {"values": [10, 15, 20]},
        "batch_size": {"values": [16, 32, 64]},
        "dropout_rate": {"values": [0, 0.5]}
    },
    "early_terminate": {
        "type": "hyperband",
        "eta": 2,
        "min_iter": 3
    }
}

@click.command()
@click.option("--experiment_name", type=str, default="demo")
@click.option("--entity", type=str, default="dragon_merger_agn")
@click.option("--n_sweeps", type=int, default=12)
@click.option(
    "--run_id",
    type=str,
    default=None,
    help="""The run id. Practically this only needs to be used
if you are resuming a previously run experiment""",
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
@click.option("--cutout_size", type=int, default=167)
@click.option("--channels", type=int, default=1)
@click.option(
    "--n_workers",
    type=int,
    default=4,
    help="""The number of workers to be used during the
data_preprocessing loading process.""",
)
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
@click.option("--num_classes", type=int, default=6)
@click.option(
    "--loss",
    type=click.Choice(
        [
            "nll",
        ],
        case_sensitive=False,
    ),
    default="nll",
    help="""The loss function to use""",
)
@click.option(
    "--crop/--no-crop",
    default=True,
    help="""If True, all images are passed through a cropping
operation before being fed into the network. Images are cropped
to the cutout_size parameter""",
)
@click.option(
    "--dropout_rate",
    type=float,
    default=None,
    help="""The dropout rate to use for all the layers in the
    model. If this is set to None, then the default dropout rate
    in the specific model is used.""",
)
def sweep_init(**kwargs):
    # Copy and log args
    args = {k: v for k, v in kwargs.items()}

    # Discover devices
    args["device"] = discover_devices()

    # Create the model given model_type
    cls = model_factory(args["model_type"])
    model_args = {
        "cutout_size": args["cutout_size"],
        "channels": args["channels"],
        "num_classes": args["num_classes"]
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

    # Select the desired transforms
    T = None
    if args["crop"]:
        T = K.CenterCrop(args["cutout_size"])

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
        )
        for k in splits
    }

    # Select the desired transforms
    T = None
    if args["crop"]:
        T = K.CenterCrop(args["cutout_size"])

    # Define the criterion
    loss_dict = {
        "nll": nn.NLLLoss(),
    }
    criterion = loss_dict[args["loss"]]

    # Log into W&B
    wandb.login()
    wandb.require("service")

    # Initializing the Sweep
    trainer_func = partial(train, model=model, datasets=datasets, criterion=criterion, args=args)
    sweep_id = wandb.sweep(sweep=sweep_config, project=args["experiment_name"])
    logging.info(f"The W&B sweep ID for this run is {sweep_id}.")

    # Multiplexing capability.
    p_args = {
        "sweep_id": sweep_id,
        "function": trainer_func,
        "project": args["experiment_name"],
        "entity": args["entity"],
        "count": (args["n_sweeps"] / args["n_workers"])
    }
    processes = []
    if args["device"] == "cpu" and args["parallel"]:  # Multiplex given N cpus
        num_agents = min(mp.cpu_count(), args["n_workers"])
        logging.info(f"Parallelizing sweeps over {num_agents} CPUs.")
        for _ in range(num_agents):
            p = mp.Process(target=wandb.agent, kwargs=p_args)
            p.start()  # Start the new child process
            processes.append(p)

        for p in processes:
            p.join()  # Thread join to wait for each to finish execution.
    elif args["device"] == "cuda" and args["parallel"]:  # Multiplexing using GPUs.
        num_agents = torch.cuda.device_count()
        devices = (torch.cuda.get_device_name(i) for i in range(num_agents))
        logging.info(f"Parallelizing sweeps over {num_agents} agents.")
        for n in devices:
            os.environ['CUDA_VISIBLE_DEVICES'] = n
            p = mp.Process(target=wandb.agent, kwargs=p_args)
            p.start()  # Start the new child process
            processes.append(p)

        for p in processes:
            p.join()  # Thread join to wait for each to finish execution.

    # Housekeeping
    sweep_path = f'{args["entity"]}/{args["experiment_name"]}/{sweep_id}'
    try:
        result = subprocess.run(['wandb', 'sweep', '--cancel', sweep_path], check=True, capture_output=True, text=True)
        logging.info(f"All runs on sweep ID {sweep_id} have terminated and sweep is now canceled.")
        logging.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"ERROR: Failed to cancel sweep {sweep_id}: {e}")

    return



def train(model, datasets, criterion, args):
    # Initializing W&B run
    with wandb.init(
        id=args["run_id"],
        resume="allow",
        group="DDP",
        entity=args["entity"],
        config={
            "num_classes": args["num_classes"],
            "architecture": "CNN"
        }
    ) as run:
        # Overriding run name if it is specified.
        if args["run_name"] is not None:
            name_str = "_".join(
                [f"{key}_{wandb.config[key]}" for key in wandb.config.keys()[2:]]
            )
            run.name = args["run_name"] + "_" + name_str

        optimizer = opt.SGD(
            model.parameters(),
            lr=wandb.config.learning_rate,
            momentum=wandb.config.momentum,
            nesterov=wandb.config.nesterov,
            weight_decay=wandb.config.weight_decay
        )

        # Create a DataLoader factory based on command-line args
        loader_factory = partial(
            get_data_loader,
            batch_size=wandb.config.batch_size,
            n_workers=args["n_workers"],
        )

        loaders = {k: loader_factory(v) for k, v in datasets.items()}
        args["splits"] = {k: len(v.dataset) for k, v in loaders.items()}

        # Write the parameters and model stats to W&B
        args = {**args, **model_stats(model)}
        wandb.log(args)

        # Set up trainer
        trainer = create_trainer(
            model, optimizer, criterion, loaders, args["device"]
        )

        # Run trainer and save model state
        trainer.run(loaders["train"], max_epochs=wandb.config.epochs)
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

    sweep_init()
