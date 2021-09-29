import torch
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


class FiLM(nn.Module):
    """
    Applies Feature-wise Linear Modulation to the incoming data as described
    in the paper `FiLM: Visual Reasoning with a General Conditioning Layer`
    (https://arxiv.org/abs/1709.07871).
    """
    def __init__(self, output_size=2048):
        super().__init__()

        self.register_parameter(name='gamma',
                                param=nn.Parameter(torch.ones(output_size)))
        self.register_parameter(name='beta',
                                param=nn.Parameter(torch.zeros(output_size)))

    def forward(self, x):
        """
        Forward pass (modulation)

        :param input: input features (N, C, *) where * is any number of dims
        :param gamma: (C,)
        :param beta: (C,)
        :return: output, modulated features (N, C, *), same shape as input
        """
        gamma = self.gamma.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        beta = self.beta.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gamma * x) + beta


class FiLMedResNet(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()

        self.resnet = models.resnet152(pretrained=True)
        layer2_input = self.resnet.layer2[0].conv1.in_channels
        layer3_input = self.resnet.layer3[0].conv1.in_channels
        layer4_input = self.resnet.layer4[0].conv1.in_channels
        self.film1 = FiLM(output_size=layer2_input)
        self.film2 = FiLM(output_size=layer3_input)
        self.film3 = FiLM(output_size=layer4_input)
        num_feats = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(num_feats, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.film3(x)

        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = x.squeeze()

        x = self.leaky_relu(self.resnet.fc(x))
        x = self.leaky_relu(self.norm(x))
        x = self.leaky_relu(self.linear(x))

        return x
