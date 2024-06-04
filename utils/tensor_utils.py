import torch
import numpy as np

def tensor_to_numpy(x):
    """Convert a torch tensor to NumPy for plotting."""
    return np.clip(x.numpy().transpose((1, 2, 0)), 0, 1)


def arsinh_normalize(X):
    """Normalize a Torch tensor with arsinh."""
    return torch.log(X + (X ** 2 + 1) ** 0.5)

def load_tensor(filename, tensors_path, as_numpy=False):
    """Load a Torch tensor from disk."""
    if not as_numpy:
        return torch.load(tensors_path / (filename + ".pt"))

    return torch.load(tensors_path / (filename + ".pt")).numpy()

