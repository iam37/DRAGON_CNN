import torch.nn as nn

from torchvision import models

from ggt.models import GGTNoGConv


class VGG(GGTNoGConv):
    """VGG model as described in Very Deep Convolutional Networks for Large-Scale
    Image Recognition (Simonyan & Zisserman, 2015), plus optional dropout."""

    def __init__(
        self,
        cutout_size,
        channels,
        n_out=1,
        dropout=0.5,
        pretrained=True,
        use_spatial_transformer=False,
    ):
        self.pretrained = pretrained
        self.use_spatial_transformer = use_spatial_transformer
        super(VGG, self).__init__(cutout_size, channels, n_out, dropout)

    def setup_featurizer(self):
        self.featurize = models.vgg16(pretrained=self.pretrained)
        self.featurize.classifier[6] = nn.Linear(4096, self.n_out)

    def setup_regression(self):
        pass

    def setup_pooling(self, input_shape=None):
        pass

    def setup_dropout(self, dropout):
        if dropout > 0:
            features = list(self.featurize.features)
            new_features = []
            for i, feature in enumerate(features):
                if isinstance(feature, nn.Conv2d) and i > 0:
                    new_features.append(nn.Dropout(p=dropout, inplace=False))
                new_features.append(feature)
            self.featurize.features = nn.Sequential(*new_features)

    def forward(self, x):
        if self.use_spatial_transformer:
            x = self.spatial_transform(x)
        x = self.featurize(x)

        return x
