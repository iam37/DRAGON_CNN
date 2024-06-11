import torch
import numpy as np
import logging

def tensor_to_numpy(x):
    """Convert a torch tensor to NumPy for plotting."""
    return np.clip(x.numpy().transpose((1, 2, 0)), 0, 1)


def arsinh_normalize(X):
    """Normalize a Torch tensor with arsinh."""
    normalized = torch.log(X + (X ** 2 + 1) ** 0.5)
    normalized[torch.isnan(normalized)] = 0  # Replace NaN values with 0
    normalized[torch.isinf(normalized)] = 255
    return normalized

def load_tensor(filename, tensors_path, device="cpu", as_numpy=False):
    """Load a Torch tensor from disk."""
    try:
        tensor = torch.load(tensors_path / (filename + ".pt"), map_location=device)
        if not as_numpy:
            return tensor
        return tensor.numpy()
    except Exception as e:
        logging.error(f"ERROR: Failed to load tensor from {filename}: {e}")
        raise

def load_tensor_to_gpu(filename, tensors_path, device, as_numpy=False):
    tensor = load_tensor(filename, tensors_path, as_numpy=False)
    return tensor.to(device)


