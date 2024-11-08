# training/feature_extractor.py

import torch
import torch.nn as nn
from torchvision import models

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])  # Remove the fully connected layer
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.resnet(x)
        x = self.flatten(x)
        return x
