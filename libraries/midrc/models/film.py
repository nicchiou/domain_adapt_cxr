"""Base FiLM module and modified ResNets with FiLM layer replacements."""
import copy

import torch
from torch import nn
from torch.nn import Module
from torchvision import models


class FiLM(Module):
    """FiLM module.

    Applies Feature-wise Linear Modulation to the incoming data as described
    in the paper `FiLM: Visual Reasoning with a General Conditioning Layer`
    (https://arxiv.org/abs/1709.07871).
    """
    def __init__(self, output_size):
        super().__init__()

        self.num_features = output_size
        self.register_parameter(name='gamma',
                                param=nn.Parameter(torch.ones(output_size)))
        self.register_parameter(name='beta',
                                param=nn.Parameter(torch.zeros(output_size)))

    def forward(self, x):
        """Forward pass (modulation)

        :param x: input features (N, C, *) where * is any number of dims
        :param gamma: (C,)
        :param beta: (C,)
        :return: output, modulated features (N, C, *), same shape as input
        """
        gamma = self.gamma.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        beta = self.beta.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gamma * x) + beta

    def extra_repr(self) -> str:
        return f'{self.num_features}'


class FiLMedResNetBeforeBlock(Module):
    """ResNet with added FiLM layers between ResNet blocks.

    ResNet with FiLM layers introduced before each new block in the ResNet
    architecture (before the convolutional layer).

    :param film_layers: list of numbers corresponding to blocks within the
                        ResNet that FiLM layers should be introduced.
    :param hidden_size: dimension of final linear layer
    :param resnet: ResNet architecture
    """
    def __init__(self, film_layers, hidden_size=1024, resnet='resnet50'):
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

        # Get size of inputs to next block
        layer1_input = self.resnet.layer1[0].conv1.in_channels
        layer2_input = self.resnet.layer2[0].conv1.in_channels
        layer3_input = self.resnet.layer3[0].conv1.in_channels
        layer4_input = self.resnet.layer4[0].conv1.in_channels
        # Initialize corresponding FiLM layers to match
        self.film1 = FiLM(output_size=layer1_input)
        self.film2 = FiLM(output_size=layer2_input)
        self.film3 = FiLM(output_size=layer3_input)
        self.film4 = FiLM(output_size=layer4_input)

        # Add in FiLM layers
        self.resnet_body = nn.ModuleList()
        if 1 in film_layers:
            self.resnet.body.append(self.film1)
        self.resnet_body.append(self.resnet.layer1)
        if 2 in film_layers:
            self.resnet_body.append(self.film2)
        self.resnet_body.append(self.resnet.layer2)
        if 3 in film_layers:
            self.resnet_body.append(self.film3)
        self.resnet_body.append(self.resnet.layer3)
        if 4 in film_layers:
            self.resnet_body.append(self.film4)
        self.resnet.body.append(self.resnet.layer4)

        num_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_feats, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x):

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        for layer in self.resnet_body:
            x = layer(x)

        x = self.resnet.avgpool(x)
        x = x.squeeze()

        x = self.leaky_relu(self.resnet.fc(x))
        x = self.linear(x)

        return x


class FiLMedResNetReplaceBN(Module):
    """ResNet with FiLM layers replacing batch normalization layers.

    ResNet with FiLM layers introduced as replacements to some of the ResNet's
    batch normalization layers.

    :param block_replace: list of custom blocks' batch norm layers to replace
    :param bn_replace: list of individual batch norm layers to replace within a
                       block
    :param hidden_size: dimension of final linear layer
    :param resnet: ResNet architecture
    :param replace_downsample: whether to replace the batch norm layer in the
                               downsampling step
    """
    def __init__(self, block_replace, bn_replace, hidden_size=1024,
                 resnet='resnet50', replace_downsample=False):
        super().__init__()

        # Must specify blocks and batch norm layers to replace
        assert bn_replace is not None
        if set(bn_replace) != {0}:
            assert block_replace is not None

        # Get ResNet backbone
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

        # Iterate through ResNet modules and replace batch norm layers with FiLM
        for name, layer in self.named_modules():
            if isinstance(layer, torch.nn.BatchNorm2d):
                # Replace the batch norm layer before any of the ResNet blocks
                if name == 'resnet.bn1':
                    if 0 in bn_replace:
                        output_size = self.resnet.bn1.weight.size(0)
                        self.resnet.bn1 = FiLM(output_size=output_size)
                # Otherwise, replace batch norm layers within the ResNet blocks
                else:
                    parsed_name = name.split('.')
                    # Layer is a numbered batch norm layer in a bottleneck unit
                    if len(parsed_name) == 4:
                        _, layer_num, bottleneck_num, bn_num = parsed_name
                        if (int(layer_num[-1]) in block_replace and
                            int(bn_num[-1]) in bn_replace):
                            output_size = self.resnet._modules[
                                layer_num]._modules[
                                    bottleneck_num]._modules[
                                        bn_num].weight.size(0)
                            self.resnet._modules[layer_num]._modules[
                                bottleneck_num]._modules[bn_num] = \
                                FiLM(output_size=output_size)
                    # Layer is a batch norm layer in the downsampling step
                    elif len(parsed_name) == 5:
                        _, layer_num, bottleneck_num, _, _ = parsed_name
                        if replace_downsample:
                            output_size = self.resnet._modules[
                                layer_num]._modules[bottleneck_num]._modules[
                                    'downsample']._modules['1'].weight.size(0)
                            self.resnet._modules[layer_num]._modules[
                                bottleneck_num]._modules['downsample']._modules[
                                    '1'] = FiLM(output_size=output_size)

        num_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_feats, hidden_size)
        self.linear = nn.Linear(hidden_size, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x):

        x = self.leaky_relu(self.resnet(x))
        x = self.linear(x)

        return x


class FiLMedResNetFineTune(Module):
    """ResNet-50 with FiLM layers replacing BN layers during fine-tuning.

    ResNet-50 with newly-initialized FiLM layers introduced only during the
    fine-tuning phase on target data. The FiLM layers are added according to
    the FiLMedResNet class passed during initialization.

    Note: Currently only supports FiLMedResNetBN

    :param pretrained: torch.nn.Module pre-trained on the source domain
    :param new_model_class: torch.nn.Module of the model to use during
                            fine-tuning on the target domain
    """
    def __init__(self, pretrained: nn.Module, new_model_class: nn.Module,
                 **kwargs):
        super().__init__()

        # Initialize new model class
        new_model = new_model_class(**kwargs)
        own_state = new_model.state_dict()

        # Avoid parameter override of FiLM layers with BN weights/bias
        other_state = pretrained.state_dict()
        if isinstance(new_model, FiLMedResNetReplaceBN):
            for name, param in other_state.items():
                if isinstance(param, nn.Parameter):
                    param = param.data
                # Check if BN parameters are found in loaded state_dict and
                # do not copy the parameter over
                try:
                    own_state[name].copy_(param)
                except KeyError:
                    assert 'bn' in name
                    continue

        self.resnet = copy.deepcopy(new_model.resnet)

        del new_model

    def forward(self, x):
        return self.resnet(x)
