import pandas as pd
import numpy as np
from pathlib import Path
from collections import Counter
import subprocess
import click
import logging


def register_voters(output_path, data_dir, model_paths, model_stems, n_classes, channels):
    logging.info(f"Registering election for {data_dir}...")

    for model_path, model_stem in zip(model_paths, model_stems):
        command = [
            'python', 'modules/inference.py',
            '--model_path', model_path,
            '--model_type', 'dragon',
            '--output_path', (output_path + model_stem),
            '--data_dir', data_dir,
            '--slug', 'balanced-dev',
            '--n_workers', "4",
            '--parallel',
            '--batch_size', "32",
            '--label_col', 'classes',
            '--n_classes', str(n_classes),
            '--normalize',
            '--dropout_rate', "0.0004",
            '--cutout_size', "94",
            '--channels', str(channels)
        ]

        # Remove empty arguments
        command = [arg for arg in command if arg]
        logging.info(f"Registering voter {model_path}...")

        # Run the command
        subprocess.run(command)


def election(output_path, data_dir):
    logging.info("Beginning election...")

    path = Path(data_dir)
    df = pd.DataFrame()  # Initialize an empty DataFrame

    index = 0
    for csv in path.glob('*inf_1.csv'):
        ndf = pd.read_csv(csv)

        # Determine if we are using file_name or object_id
        id_col = "file_name" if "file_name" in ndf.columns else "object_id"

        # Append identifier column if not already present in the main df
        if id_col not in df.columns:
            df[id_col] = ndf[id_col]

        # Add predicted labels as new columns
        voter_column = f"voter_{index}"
        voter_confidence = f"voter_{index}_conf"

        df[voter_column] = ndf["predicted_labels"]
        df[voter_confidence] = ndf["predicted_confidence"]

        index += 1

    # Combined results
    logging.info(f"Election counts saved to {output_path}/combined_results.csv")
    df.to_csv(Path(output_path) / 'combined_results.csv', index=False)


def congress(path, csv='combined_results.csv'):
    df = pd.read_csv(path + '/combined_results.csv')

    # Determine the identifier column
    id_col = "file_name" if "file_name" in df.columns else "object_id"

    congress = pd.DataFrame(columns=[id_col, "voted_class", "num_voters", "total_voters", "average_confidence",
                                     "weighted_confidence"])

    logging.info("Sending results for certification...")

    num_voters = 0
    for header in df.columns:
        if "voter" in header and not "conf" in header:
            num_voters += 1

    if num_voters == 0:
        return

    # Calculations of optimism
    total_counts = list()

    # 1. Safely find all unique labels across ALL voters
    all_labels = set()
    for index in range(num_voters):
        all_labels.update(df[f"voter_{index}"].unique())

    min_counts = {k: float('inf') for k in all_labels}
    max_counts = {k: 0 for k in all_labels}

    for index in range(num_voters):
        counts = df[f"voter_{index}"].value_counts()
        total_counts.append(counts)
        for key in counts.index:
            min_counts[key] = min(counts[key], min_counts[key])
            max_counts[key] = max(counts[key], max_counts[key])

    # First round voting
    for _, row in df.iterrows():
        voter_vals = Counter([row[f"voter_{index}"] for index in range(num_voters)])
        confidence_vals = [row[f"voter_{index}_conf"] for index in range(num_voters)]
        majority, maj_count = voter_vals.most_common(1)[0]

        voted_class = majority

        # If there is another one
        if len(voter_vals) > 1:
            second, second_count = voter_vals.most_common(2)[1]
            if maj_count - 1 <= second_count <= maj_count:
                voted_class = -1

        # Average confidence calculation
        avg_confidence = sum(confidence_vals) / len(confidence_vals)

        # Weighted voter score!
        # 2. Use .get(majority, 0) to prevent KeyErrors if a voter never predicted the majority class
        optimisms = [total_counts[index].get(majority, 0) / max_counts[majority] for index in range(num_voters)]
        weighted_probs = [confidence_vals[i] * optimisms[i] for i in range(num_voters)]

        # Guard against division by zero if sum(optimisms) is 0
        weighted_denom = sum(optimisms)
        weighted_confidence = sum(weighted_probs) / weighted_denom if weighted_denom > 0 else 0

        congress.loc[len(congress)] = {
            id_col: row[id_col],  # Assuming you are using the dynamic id_col from the previous fix
            "voted_class": voted_class,
            "num_voters": maj_count,
            "total_voters": num_voters,
            "average_confidence": avg_confidence,
            "weighted_confidence": weighted_confidence
        }

    logging.info("Congressional voting completed...")
    congress.to_csv(path + '/congress.csv', index=False)


@click.command()
@click.option(
    "--data_dirs",
    type=str,
    default="tang_candidates,merger_test",
    help="""Enter the target data_dir separated by commas""",
    required=True
)
@click.option(
    "--model_folder",
    type=click.Path(exists=True),
    default="good_models/voters",
    required=True
)
@click.option("--n_classes", type=int, default=6)
@click.option("--channels", type=int, default=1)
def run_elections(**kwargs):
    logging.info("Getting voters (models) to the polls (classifications)...")

    # Copy and log args
    args = {k: v for k, v in kwargs.items()}

    data_dirs = args["data_dirs"].split(",")
    output_paths = [data_dir + '/' for data_dir in data_dirs]
    model_path = Path(args["model_folder"])

    model_paths = [fl.as_posix() for fl in model_path.glob('*.pt')]
    model_stems = [fl.stem for fl in model_path.glob('*.pt')]

    for output_path, data_dir in zip(output_paths, data_dirs):
        register_voters(
            output_path, data_dir, model_paths, model_stems,
            n_classes=args["n_classes"], channels=args["channels"]
        )
        election(output_path, data_dir)
        congress(data_dir)

        logging.info(f"Election finished for {data_dir}.")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    logging.info("Running multi-party elections...")
    run_elections()