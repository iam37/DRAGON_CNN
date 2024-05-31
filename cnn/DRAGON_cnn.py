import numpy as np

import torch
import torch.nn as nn

class DRAGON(nn.Module):
    def __init__(self):
        super(DRAGON, self).__init__()

    def forward(self, x):
        return x