"""ResNet wrapper modules for torchvision models."""
from torch import nn
from torch.nn import Module
from torchvision import models


class ResNetClassifier(Module):
    """Loads an ImageNet pre-trained ResNet.

    Replaces the final linear layer for a binary prediction task.

    Args:
      resnet: ResNet architecture to use
    """
    def __init__(self, hidden_size: int = 1024, resnet: str = 'resnet50'):
        super().__init__()

        if resnet == 'resnet18':
            self.resnet = models.resnet18(pretrained=True)
        elif resnet == 'resnet34':
            self.resnet = models.resnet34(pretrained=True)
        elif resnet == 'resnet50':
            self.resnet = models.resnet50(pretrained=True)
        elif resnet == 'resnet101':
            self.resnet = models.resnet101(pretrained=True)
        elif resnet == 'resnet152':
            self.resnet = models.resnet152(pretrained=True)
        num_feats = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(num_feats, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        x = self.leaky_relu(self.resnet(x))
        x = self.linear(x)

        return x
