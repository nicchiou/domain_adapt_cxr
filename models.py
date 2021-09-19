from torch import nn
from torchvision import models


class ResNetClassifier(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()

        self.resnet = models.resnet152(pretrained=True)
        num_feats = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(num_feats, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        x = self.leaky_relu(self.resnet(x))
        x = self.leaky_relu(self.norm(x))
        x = self.leaky_relu(self.linear(x))

        return x
