
import numpy as np

import torch
import torch.nn as nn

class DRAGON_260(nn.Module):
    def __init__(self, cutout_size=260, channels=1, num_classes=6):
        super(DRAGON_260, self).__init__()
        self.cutout_size = cutout_size
        self.channels = channels
        self.expected_input_shape = (
            16,
            self.channels,
            self.cutout_size,
            self.cutout_size,
        )

        self.num_classes = num_classes

        self.layer1 = nn.Sequential(
            nn.Conv2d(self.channels, 64, kernel_size=3, padding='same'),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 96, kernel_size=3, padding='same'),
            nn.BatchNorm2d(96),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(96, 128, kernel_size=3, padding='same'),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding='same'),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1),
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, padding='same'),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=1)
        )

        self.layer6 = nn.Sequential(
            nn.Conv2d(384, 512, kernel_size=3, padding='same'),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2)
        )

        self.fc1 = nn.Linear(512, 1024)  # Adjusted input size to match flattened output
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        # print('Layer1 output:', out.shape)
        out = self.layer2(out)
        # print('Layer2 output:', out.shape)
        out = self.layer3(out)
        # print('Layer3 output:', out.shape)
        out = self.layer4(out)
        # print('Layer4 output:', out.shape)
        out = self.layer5(out)
        # print('Layer5 output:', out.shape)
        out = self.layer6(out)
        # print('Layer6 output:', out.shape)

        # Flatten the output tensor
        out = out.view(out.size(0), -1)
        # print('Flattened output:', out.shape)

        # Fully connected layers
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)

        return out
