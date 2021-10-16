import torch
from pathlib import Path

from .ggt import GGT, GGTNoGConv
from .vgg import vgg16, vgg16_w_stn_drp


def model_stats(model):
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(trainable_params=n_params)


def model_factory(modeltype):
    if modeltype.lower() == "ggt":
        return GGT
    elif modeltype.lower() == "ggt_no_gconv":
        return GGTNoGConv
    elif modeltype.lower() == "vgg16_w_stn_drp" or "vgg16_w_stn":
        return vgg16_w_stn_drp
    elif modeltype.lower() == "vgg16":
        return vgg16
    else:
        raise ValueError("Model type {} does not exist.".format(modeltype))


def save_trained_model(model, slug):
    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{slug}.pt"
    torch.save(model.state_dict(), dest)
    return dest
