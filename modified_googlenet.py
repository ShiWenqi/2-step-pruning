import torch
from models import *
import torch.nn as nn


class ModifiedGoogleNet(torch.nn.Module):
    def __init__(self):
        super(ModifiedGoogleNet, self).__init__()
        model = GoogLeNet()
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
        self.features = model
        self.classifier = nn.Sequential(nn.Linear(1024,10))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x