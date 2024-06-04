import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from ignite.engine import Engine, Events
from ignite.metrics import Accuracy, Recall, Precision, Fbeta, confusion_matrix, ConfusionMatrix, EpochMetric
from torchvision import models


class ResNet(nn.Module):
    def __init__(
            self,
            channels,
            n_classes=6,
            pretrained=True,
            dropout=False,
            dropout_rate=0.05
    ):
        super(ResNet, self).__init__()
        self.channels = channels
        self.n_classes = n_classes
        self.pretrained = pretrained

        self.model = models.resnet50(pretrained=self.pretrained)
        self.model.conv1 = nn.Conv2d(self.channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        for param in self.model.parameters():
            param.requires_grad = False

        self.model.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.model.fc = nn.Sequential(
            nn.BatchNorm1d(num_features=2048, eps=1e-05, momentum=0.1),
            nn.Dropout(p=dropout_rate if dropout else 0),
            nn.Linear(2048, 16),
            nn.LeakyReLU(),
            nn.Linear(16, self.n_classes)
        )

    def forward(self, x):
        x = torch.nan_to_num(x, nan=0, posinf=1)
        output = self.model(x)

        if self.n_out < 2:
            output = torch.flatten(output)

        return output

