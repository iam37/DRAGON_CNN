# -*- coding: utf-8 -*-
import click
import logging

import torch
import torch.nn as nn

from tqdm import tqdm

from data_preprocessing import FITSDataset, get_data_loader
import kornia.augmentation as K

from cnn import model_factory
from utils import (
    discover_devices,
    enable_dropout,
    specify_dropout_rate,
    load_data_dir
)


def predict(
    model_path,
    dataset,
    cutout_size,
    channels,
    parallel=False,
    batch_size=256,
    n_workers=1,
    num_classes=6,
    model_type="dragon",
    mc_dropout=False,
    dropout_rate=None,
    apply_softmax=True
):
    """Using the model defined in model path, return the output values for
    the given set of images"""

    # Discover devices
    device = discover_devices()

    # Declare the model given model_type
    cls = model_factory(model_type)
    model_args = {
        "cutout_size": cutout_size,
        "channels": channels,
        "num_classes": num_classes
    }

    if "drp" in model_type.split("_"):
        logging.info(
            "Using dropout rate of {} in the model".format(dropout_rate)
        )
        model_args["dropout"] = "True"

    model = cls(**model_args)
    model = nn.DataParallel(model) if parallel else model
    model = model.to(device)

    # Changing the dropout rate if specified
    if dropout_rate is not None:
        specify_dropout_rate(model, dropout_rate)

    # Load the model
    logging.info("Loading model...")
    if device == "cpu":
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
    else:
        model.load_state_dict(torch.load(model_path))

    # Create a data_preprocessing loader
    loader = get_data_loader(
        dataset,
        batch_size=batch_size,
        n_workers=n_workers,
        shuffle=False,
    )

    logging.info("Performing predictions...")
    yh = []
    model.eval()

    # Enable Monte Carlo dropout if requested
    if mc_dropout:
        logging.info("Activating Monte Carlo dropout...")
        enable_dropout(model)

    with torch.no_grad():
        for data in tqdm(loader):
            X, _ = data
            X = X.to(device)
            outputs = model(X)
            if apply_softmax:
                outputs = nn.functional.softmax(outputs, dim=1)
            yh.append(outputs)

    yh = torch.cat(yh)

    # Get the predicted class indices
    predicted_confs, predicted_labels = torch.max(yh, 1)

    return predicted_labels.cpu().numpy(), predicted_confs.cpu().numpy()


@click.command()
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
@click.option("--model_path", type=click.Path(exists=True), required=True)
@click.option("--output_path", type=click.Path(writable=True), required=True)
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option("--cutout_size", type=int, default=167)
@click.option("--channels", type=int, default=3)
@click.option(
    "--slug",
    type=str,
    required=True,
    help="""This specifies which slug (balanced/unbalanced
              xs, sm, lg, dev) is used to perform predictions on.""",
)
@click.option("--split", type=str, required=True, default="test")
@click.option(
    "--normalize/--no-normalize",
    default=True,
    help="""The normalize argument controls whether or not, the
              loaded images will be normalized using the arsinh function""",
)
@click.option("--batch_size", type=int, default=256)
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
    "--label_col",
    type=str,
    default="classes",
    help="""Enter the label column(s) separated by commas. Note
    that you should pass the exactly same argument for label_col
    as was used during the training phase (of the model being used
    for inference). """,
)
@click.option(
    "--mc_dropout/--no-mc_dropout",
    default=True,
    help="""Turn on Monte Carlo dropout during inference.""",
)
@click.option(
    "--n_runs",
    type=int,
    default=1,
    help="""The number of times to run inference. This is helpful
    when usng mc_dropout""",
)
@click.option("--n_classes", type=int, default=6)
@click.option(
    "--ini_run_num",
    type=int,
    default=1,
    help="""The number of the first run. i.e. the output csv files
    are named as (inf_run_num+iteration_number).csv""",
)
@click.option(
    "--dropout_rate",
    type=float,
    default=None,
    help="""The dropout rate to use for all the layers in the
    model. If this is set to None, then the default dropout rate
    in the specific model is used. This option should only be
    used when you have used a non-default dropout rate during
    training and have set --mc_dropout to True. The rate should
    be set equal to the rate used during training.""",
)
@click.option(
    "--crop /--no-crop",
    default=True,
    help="""If True, the images are passed through a cropping transformation
to ensure proper cutout size""",
)
@click.option(
    "--labels/--no-labels",
    default=True,
    help="""If True, this means you have labels available for the dataset.
    If False, this means that you have no labels available and want to do
    pure inference using a pre-trained model.""",
)
def main(
    model_path,
    output_path,
    data_dir,
    cutout_size,
    channels,
    parallel,
    slug,
    split,
    normalize,
    batch_size,
    n_workers,
    label_col,
    model_type,
    mc_dropout,
    dropout_rate,
    crop,
    n_runs,
    n_classes,
    ini_run_num,
    labels,
):

    logging.info(
        """Performing pure inference without labels. Using
            column names to infer number of expected outputs.
            Split and Slug values entered will be ignored and
            info.csv will be used."""
    )
    split = None
    slug = None

    # Create label cols array
    label_col_arr = label_col.split(",")

    # Transforming the dataset to the proper cutout size
    T = None
    if crop:
        T = K.CenterCrop(cutout_size)

    # Test

    # Load the data_preprocessing and create a data_preprocessing loader
    logging.info("Loading images to device...")
    dataset = FITSDataset(
        data_dir,
        slug=slug,
        normalize=normalize,
        split=split,
        cutout_size=cutout_size,
        channels=channels,
        label_col=label_col_arr,
        transforms=T,
        load_labels=False
    )

    for run_num in range(ini_run_num, n_runs + ini_run_num):

        logging.info(f"Running inference run {run_num}")

        # Make predictions
        preds, cis = predict(
            model_path,
            dataset,
            cutout_size,
            channels,
            parallel=parallel,
            batch_size=batch_size,
            n_workers=n_workers,
            num_classes=n_classes,
            model_type=model_type,
            mc_dropout=mc_dropout,
            dropout_rate=dropout_rate,
        )

        # Write a CSV of predictions
        catalog = load_data_dir(data_dir, slug, split)
        catalog["predicted_labels"] = preds
        catalog["predicted_confidence"] = cis  # Writing probabilities.

        cat_path = output_path + f"inf_{run_num}.csv"
        logging.info(f"Catalog saved to {cat_path}")
        catalog.to_csv(cat_path, index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
