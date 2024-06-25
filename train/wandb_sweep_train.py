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
from train import create_trainer, create_transfer_learner
from utils import discover_devices, specify_dropout_rate

# Global Sweep Configuration. This also effects early stopping
# for bad runs!
sweep_config = {
    "method": "bayes",
    "metric": {"goal": "maximize", "name": "devel_accuracy"},
    "parameters": {
        "learning_rate": {"values": [0.001, 0.0001, 0.0005]},
        "momentum": {"values": [1e-4, 1e-5, 1e-6]},
        "nesterov": {"values": [True, False]},
        "weight_decay": {"values": [1e-1, 1e-2, 1e-3]},
        "epochs": {"values": [10, 15, 20]},
        "batch_size": {"values": [16, 32, 64]},
        "dropout_rate": {"values": [0, 0.2, 0.3, 0.4, 0.5]},
        "scheduler": {"values": [True, False]}
    },
    "early_terminate": {
        "type": "hyperband",
        "eta": 2,
        "min_iter": 3,
        "strict": True  # Corrected
    }
}

import random
import numpy as np

def initialize_and_run_agent(device, p_args):
    if device is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device)
    reset_wandb_env()  # Reset W&B environment
    wandb.agent(**p_args)


def free_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_model_and_optimizer(model, optimizer):
    del model
    del optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def reset_wandb_env():
    logging.info("Resetting W&B environment to ensure separation.")
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]

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
            "dragon",
            "resnet"
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
@click.option("--n_classes", type=int, default=6)
@click.option(
    "--loss",
    type=click.Choice(
        [
            "nll",
            "ce"
        ],
        case_sensitive=False,
    ),
    default="ce",
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
    "--force_reload/--no_force_reload",
    default=False,
)
@click.option(
    "--train/--transfer_learn",
    default=True,
    help="""Specifies whether you wish to do transfer learning. If transfer learning,
    you must specify model path in the model_state argument."""
)
def sweep_init(**kwargs):
    # Copy and log args
    args = {k: v for k, v in kwargs.items()}

    # Discover devices
    args["device"] = discover_devices()

    # Create the model given model_type
    cls = model_factory(args["model_type"])

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
            force_reload=args["force_reload"],
            num_classes=args["n_classes"]
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
        "ce": nn.CrossEntropyLoss()
    }
    criterion = loss_dict[args["loss"]]

    # Log into W&B
    reset_wandb_env()  # Initial reset of W&B environment.
    wandb.login()
    wandb.require("service")

    # Initializing the Sweep
    trainer_func = partial(train, model_cls=cls, datasets=datasets, criterion=criterion, args=args)
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
        num_agents = 1 # torch.cuda.device_count()
        logging.info(f"Parallelizing sweeps over {num_agents} agents.")

        for i in range(num_agents):
            p = mp.Process(target=initialize_and_run_agent, args=(0, p_args))
            p.start()  # Start the new child process
            processes.append(p)

        for p in processes:
            p.join()  # Thread join to wait for each to finish execution.

    # Housekeeping
    sweep_path = f'{args["entity"]}/{args["experiment_name"]}/{sweep_id}'
    sweep_list = ['wandb', 'sweep', '--cancel', sweep_path]
    try:
        result = subprocess.run(" ".join(sweep_list), shell=True)
        logging.info(f"All runs on sweep ID {sweep_id} have terminated and sweep is now canceled.")
        logging.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"ERROR: Failed to cancel sweep {sweep_id}: {e}")

    return


def train(model_cls, datasets, criterion, args):
    # Initializing W&B run
    with wandb.init(
        id=args["run_id"],
        resume="allow",
        group="DDP",
        entity=args["entity"],
        config={
            "num_classes": args["n_classes"],
            "architecture": "CNN"
        },
        reinit=True
    ) as run:
        # Overriding run name if it is specified.
        name_str = "_".join(
            [f"{key}_{wandb.config[key]}" for key in wandb.config.keys()[2:]]
        )
        if args["run_name"] is not None:
            run.name = args["run_name"] + "_" + name_str
        else:
            run.name = name_str

        model_args = {
            "cutout_size": args["cutout_size"],
            "channels": args["channels"],
            "num_classes": args["n_classes"]
        }

        logging.info("Reinitializing model.")
        model = model_cls(**model_args)
        model = nn.DataParallel(model) if args["parallel"] else model
        model = model.to(args["device"])

        # Chnaging the default dropout rate if specified
        specify_dropout_rate(model, wandb.config.dropout_rate)

        if args["model_state"]:
            logging.info(f'Loading model from {args["model_state"]}...')
            if args["device"] == "cpu":
                model.load_state_dict(torch.load(args["model_state"], map_location="cpu"))
            else:
                model.load_state_dict(torch.load(args["model_state"]))

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

        loaders = {k: loader_factory(v, shuffle=(k == 'train')) for k, v in datasets.items()}
        args["splits"] = {k: len(v.dataset) for k, v in loaders.items()}

        # Write the parameters and model stats to W&B
        args = {**args, **model_stats(model)}
        wandb.log(args)
        wandb.watch(model, log_freq=1)

        # Set up trainer
        if args["train"]:
            logging.info("Creating trainer...")
            trainer = create_trainer(
                model, optimizer, criterion, loaders, args["device"], wandb.config.scheduler
            )
        else:
            logging.info("Creating trainer and freezing layers for transfer learning...")
            trainer = create_transfer_learner(
                model, optimizer, criterion, loaders, args["device"], wandb.config.scheduler
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

        # Resetting model.
        reset_model_and_optimizer(model, optimizer)

        # Finish the W&B run!
        wandb.finish()
        free_gpu_memory()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Setting multiprocess spawn method.
    if torch.cuda.is_available():
        mp.set_start_method('spawn')
        logging.info("Setting multiprocessing start method to 'spawn'.")

    sweep_init()
