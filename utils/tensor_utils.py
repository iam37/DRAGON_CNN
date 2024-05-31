import torch
import numpy as np

def tensor_to_numpy(x):
    """Convert a torch tensor to NumPy for plotting."""
    return np.clip(x.numpy().transpose((1, 2, 0)), 0, 1)


def load_tensor(filename, tensors_path, as_numpy=True):
    """Load a Torch tensor from disk."""
    return torch.load(tensors_path / (filename + ".pt"))

