# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import numpy as np
from itertools import chain, zip_longest
import pandas as pd
from sklearn.utils import resample

split_types = dict(
    xs=dict(train=0.027, devel=0.003, test=0.970),
    sm=dict(train=0.045, devel=0.005, test=0.950),
    md=dict(train=0.090, devel=0.010, test=0.900),
    lg=dict(train=0.200, devel=0.050, test=0.750),
    xl=dict(train=0.450, devel=0.050, test=0.500),
    dev=dict(train=0.700, devel=0.150, test=0.150),
    dev2=dict(train=0.700, devel=0.050, test=0.250),
)


def interleave(L):
    return [x for x in chain(*zip_longest(*L)) if x is not None]


def balance_dataset(df, label_col):
    """Balance the dataset by oversampling the underrepresented classes."""
    classes = df[label_col].unique()
    max_count = df[label_col].value_counts().max()

    balanced_df = pd.DataFrame()
    for cls in classes:
        cls_samples = df[df[label_col] == cls]
        balanced_cls_samples = resample(cls_samples, replace=True, n_samples=max_count, random_state=0)
        balanced_df = pd.concat([balanced_df, balanced_cls_samples])

    return balanced_df.sample(frac=1).reset_index(drop=True)


def make_splits(x, weights, label_col):
    balanced_df = balance_dataset(x, label_col)
    total_size = len(balanced_df)
    indices = np.arange(total_size)
    np.random.shuffle(indices)

    splits = dict()
    prev_index = 0
    for k, v in weights.items():
        next_index = prev_index + int(total_size * v)
        splits[k] = balanced_df.iloc[indices[prev_index:next_index]]
        prev_index = next_index
    return splits


@click.command()
@click.option("--data_dir", type=click.Path(exists=True), required=True)
@click.option("--target_metric", type=str, default="classes")
@click.option("--info_name", type=str, default="info.csv")
def main(data_dir, target_metric, info_name):
    """Generate train/devel/test splits from the dataset provided."""
    data_dir = Path(data_dir)
    splits_dir = data_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_dir / info_name)
    df = df.sample(frac=1, random_state=0)

    for split_type in split_types.keys():
        splits = make_splits(df, split_types[split_type], label_col=target_metric)
        split_slug = f"balanced-{split_type}"
        for k, v in splits.items():
            v.to_csv(splits_dir / f"{split_slug}-{k}.csv", index=False)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
