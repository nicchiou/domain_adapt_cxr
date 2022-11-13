import copy

import torch
from torch import nn
from torchvision import models


class FiLM(nn.Module):
    """FiLM module.

    Applies Feature-wise Linear Modulation to the incoming data as described
    in the paper `FiLM: Visual Reasoning with a General Conditioning Layer`
    (https://arxiv.org/abs/1709.07871).
    """
    def __init__(self, output_size):
        super().__init__()

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


class FiLMedResNetBeforeBlock(nn.Module):
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


class FiLMedResNetReplaceBN(nn.Module):
    """ResNet with FiLM layers replacing batch normalization layers.

    ResNet with FiLM layers introduced as replacements to some of the ResNet's
    batch normalization layers.

    :param replace_mode: sparse (replace the last bottleneck unit's last BN
                         (bn3) in each block), all (replace all bottleneck
                         units' last BN (bn3) layer), or custom (replace
                         certain last BN (bn3) layers in consecutive
                         bottlneck units within block 3)
    :param replace_layers: list of custom blocks' BN layers to replace
    :param bn_replace: list of custom bottleneck units' BN layers to replace
                       within a block
    :param hidden_size: dimension of final linear layer
    :param resnet: ResNet architecture
    """
    def __init__(self, replace_mode, replace_layers, bn_replace,
                 hidden_size=1024, resnet='resnet50'):
        super().__init__()

        # Must specify blocks and bottleneck units to replace
        if replace_mode == 'custom':
            assert replace_layers and bn_replace
        else:
            assert not replace_layers

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
        layer2_input = self.resnet.layer2[0].conv1.in_channels
        layer3_input = self.resnet.layer3[0].conv1.in_channels
        layer4_input = self.resnet.layer4[0].conv1.in_channels

        if replace_mode == 'sparse':
            # Replace layer1 final bn3
            self.resnet.layer1._modules['2']._modules['bn3'] = \
                FiLM(output_size=layer2_input)
            # Replace layer2 final bn3
            self.resnet.layer2._modules['3']._modules['bn3'] = \
                FiLM(output_size=layer3_input)
            # Replace layer3 final bn3
            self.resnet.layer3._modules['5']._modules['bn3'] = \
                FiLM(output_size=layer4_input)

        elif replace_mode == 'all':
            for name, layer in self.named_modules():
                # Replace all batch norm layers before a ReLU with FiLM
                if isinstance(layer, torch.nn.BatchNorm2d):
                    try:  # Check if batch norm is in one of the main layers
                        _, layer_num, block_num, bn_num = name.split('.')
                        if int(bn_num[-1]) in bn_replace:
                            output_size = self.resnet._modules[
                                layer_num]._modules[
                                block_num]._modules[bn_num].weight.size(0)
                            self.resnet._modules[layer_num]._modules[
                                block_num]._modules[bn_num] = \
                                FiLM(output_size=output_size)
                    except ValueError:  # Catch exceptions from parsing name
                        continue

        elif replace_mode == 'custom':
            for name, layer in self.named_modules():
                # Replace batch norm layers before a ReLU that match the
                # specified pattern.
                # User specifies main layer, or sub-layer deliniations for
                # main layer 3 (i.e. 3.1, 3.2, 3.3).
                if isinstance(layer, torch.nn.BatchNorm2d):
                    try:
                        _, layer_num, block_num, bn_num = name.split('.')
                        if int(layer_num[-1]) in \
                                [int(x) for x in replace_layers] and \
                                int(bn_num[-1]) in bn_replace:
                            if 3.1 in replace_layers and \
                                    int(block_num) >= 0 and \
                                    int(block_num) < 2:
                                output_size = self.resnet._modules[
                                    layer_num]._modules[
                                    block_num]._modules[bn_num].weight.size(0)
                                self.resnet._modules[layer_num]._modules[
                                    block_num]._modules[bn_num] = \
                                    FiLM(output_size=output_size)
                            if 3.2 in replace_layers and \
                                    int(block_num) >= 2 and \
                                    int(block_num) < 4:
                                output_size = self.resnet._modules[
                                    layer_num]._modules[
                                    block_num]._modules[bn_num].weight.size(0)
                                self.resnet._modules[layer_num]._modules[
                                    block_num]._modules[bn_num] = \
                                    FiLM(output_size=output_size)
                            if 3.3 in replace_layers and \
                                    int(block_num) >= 4 and \
                                    int(block_num) < 6:
                                output_size = self.resnet._modules[
                                    layer_num]._modules[
                                    block_num]._modules[bn_num].weight.size(0)
                                self.resnet._modules[layer_num]._modules[
                                    block_num]._modules[bn_num] = \
                                    FiLM(output_size=output_size)
                            if float(layer_num[-1]) in replace_layers:
                                output_size = self.resnet._modules[
                                    layer_num]._modules[
                                    block_num]._modules[bn_num].weight.size(0)
                                self.resnet._modules[layer_num]._modules[
                                    block_num]._modules[bn_num] = \
                                    FiLM(output_size=output_size)
                    except ValueError:
                        continue

        num_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_feats, 1)

    def forward(self, x):
        return self.resnet(x)


class FiLMedResNetFineTune(nn.Module):
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
        if isinstance(new_model, FiLMedResNetBN):
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


class FiLMedResNetInitialLayers(nn.Module):
    """ResNet-50 with FiLM layer replacing initial BN layer.

    ResNet-50 with FiLM layers introduced near the input layer, replacing the
    first batch norm layer (bn1).
    """
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet50(pretrained=True)

        out_channels = self.resnet.conv1.out_channels
        self.film = FiLM(output_size=out_channels)

        num_feats = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_feats, 1)

    def forward(self, x):

        x = self.resnet.conv1(x)
        x = self.film(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        x = self.resnet.layer1(x)
        x = self.resnet.layer2(x)
        x = self.resnet.layer3(x)
        x = self.resnet.layer4(x)
        x = self.resnet.avgpool(x)
        x = x.squeeze()

        x = self.resnet.fc(x)

        return x
