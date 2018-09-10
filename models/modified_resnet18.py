import torchvision.models as models
import torch
import torch.nn as nn
from torch.autograd import Variable


class ModifiedResNet18Model(torch.nn.Module):
    def __init__(self):
        super(ModifiedResNet18Model, self).__init__()
        model = models.resnet18(pretrained=True)
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
        self.features = model

        self.fc = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.views(x.size(0), -1)
        x = self.fc(x)
        return x
