import pandas as pd
from pathlib import Path
import torch
import torch.nn.functional as F


def load_data_dir(data_dir, slug=None, split=None):
    """Loads and returns pandas dataframe"""

    data_dir = Path(data_dir)

    if split:
        catalog = data_dir / f"splits/{slug}-{split}.csv"
    else:
        catalog = data_dir / "info.csv"

    return pd.read_csv(catalog)


# Test

def pad_collate_fn(batch):
    # Extract the tensors from the tuples in the batch
    tensors = [item[0] for item in batch]  # Assuming tensor is the first element in the tuple
    labels = [item[1] for item in batch]   # Assuming label is the second element in the tuple

    # Find the maximum length
    max_len = max(tensor.size(0) for tensor in tensors)

    # Pad all tensors to the maximum length
    padded_tensors = [F.pad(tensor, (0, max_len - tensor.size(0))) for tensor in tensors]

    # Stack the padded tensors and convert labels to tensor
    padded_batch = torch.stack(padded_tensors)
    labels = torch.tensor(labels)

    # Return a tuple of the padded batch and labels
    return padded_batch, labels
