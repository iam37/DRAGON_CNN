import torch
from pathlib import Path
from .DRAGON_cnn import DRAGON
from .resnet import ResNet


def model_stats(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(trainable_params=n_params)


def model_factory(model_name):
    if model_name.lower() == 'dragon':
        return DRAGON
    elif model_name.lower() == 'resnet':
        return ResNet
    else:
        raise ValueError(f"Invalid model name: {model_name}")


def save_trained_model(model, slug):
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{slug}.pt"
    torch.save(model.state_dict(), dest)
    return dest


__all__: ["model_factory", "DRAGON", "ResNet"]


