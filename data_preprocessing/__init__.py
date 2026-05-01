from .dataset import FITSDataset
from torch.utils.data import DataLoader


def get_data_loader(dataset, batch_size, n_workers, shuffle=True, sampler=None):
    # IMPORTANT: If a sampler is provided, shuffle MUST be False.
    # The DistributedSampler handles the shuffling internally.
    if sampler is not None:
        shuffle = False

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=n_workers,
        sampler=sampler,  # Pass the sampler here
        pin_memory=True,  # Helps with GPU data transfer
        prefetch_factor=8,
        persistent_workers=(n_workers > 0)
    )


__all__ = ["FITSDataset", "get_data_loader"]