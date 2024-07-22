import os
import sys
import pandas as pd
from tqdm import tqdm
from astropy.io import fits
import operator
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib
import subprocess
from collections import Counter
import subprocess
import click

import logging

def register_voters(output_path, data_dir, model_paths, model_stems):
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
            '--n_classes', "6",
            '--normalize',
            '--dropout_rate', "0.0004",
            '--cutout_size', "94",
            '--channels', "1"
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

        # Append file names if not already present
        if "file_name" not in df.columns:
            df["file_name"] = ndf["file_name"]
        
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
    congress = pd.DataFrame(columns=["file_name", "voted_class", "num_voters", "total_voters", "average_confidence",
                                     "weighted_confidence"])

    logging.info("Sending results for certification...")
    
    num_voters = 0
    for header in df.columns:
        if "voter" in header and not "conf" in header:
            num_voters += 1
    
    if num_voters == 0:
        return
            
    # Calculations of optimism
    labels = np.unique(df["voter_0"])   
    min_counts = {k: float('inf') for k in range(max(labels) + 1)}
    max_counts = {k: 0 for k in range(max(labels) + 1)}
    total_counts = list()
    
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
        optimisms = [total_counts[index][majority] / max_counts[majority] for index in range(num_voters)]
        weighted_probs = [confidence_vals[i] * optimisms[i] for i in range(num_voters)]
        weighted_denom = sum(optimisms)
        weighted_confidence = sum(weighted_probs) / weighted_denom
        
        congress.loc[len(congress)] = {
            "file_name": row["file_name"], 
            "voted_class": voted_class,
            "num_voters": maj_count,
            "total_voters": num_voters,
            "average_confidence": avg_confidence,
            "weighted_confidence": weighted_confidence
        }
        
    logging.info("Congressional voting completed...")
    congress.to_csv(path + '/congress.csv')
    
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
        register_voters(output_path, data_dir, model_paths, model_stems)
        election(output_path, data_dir)
        congress(data_dir)
        
        logging.info(f"Election finished for {data_dir}.")

if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    
    logging.info("Running multi-party elections...")
    run_elections()
    