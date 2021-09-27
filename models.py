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
    def __init__(self):
        super().__init__()

    def forward(self, x, gamma, beta):
        """
        Forward pass (modulation)

        :param input: input features (N, C, *) where * is any number of dims
        :param gamma: (N, C)
        :param beta: (N, C)
        :return: output, modulated features (N, C, *), same shape as input
        """
        gamma = gamma.unsqueeze(2).unsqueeze(3).expand_as(x)
        beta = beta.unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gamma * x) + beta


class FiLMGenerator(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()

        self.resnet = models.resnet34(pretrained=True)
        self.clf = models.resnet152()
        num_feats = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(num_feats, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.gamma_func = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, self.clf.layer4[0].conv1.in_channels)
        )
        self.beta_func = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Linear(hidden_size, self.clf.layer4[0].conv1.in_channels)
        )
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        del self.clf

    def forward(self, x):

        x = self.leaky_relu(self.resnet(x))
        x = self.leaky_relu(self.norm(x))
        gammas = self.leaky_relu(self.gamma_func(x))
        betas = self.leaky_relu(self.beta_func(x))

        return gammas, betas


class FiLMedResNet(nn.Module):
    def __init__(self, hidden_size=1024):
        super().__init__()

        self.resnet = models.resnet152(pretrained=True)
        self.film = FiLM()
        num_feats = self.resnet.fc.in_features

        self.resnet.fc = nn.Linear(num_feats, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x, gamma, beta):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.film(x, gamma, beta)

        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = x.squeeze()

        x = self.leaky_relu(self.resnet.fc(x))
        x = self.leaky_relu(self.norm(x))
        x = self.leaky_relu(self.linear(x))

        return x
