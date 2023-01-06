"""Base FiLM module and modified ResNets with FiLM layer replacements."""
import copy
from typing import List

import torch
from models.resnet import ResNetClassifier
from torch import nn
from torch.nn import Module
from utils.constants import RESNET_BLOCK_SIZES


class FiLM(Module):
    """FiLM module.

    Applies Feature-wise Linear Modulation to the incoming data as described
    in the paper `FiLM: Visual Reasoning with a General Conditioning Layer`
    (https://arxiv.org/abs/1709.07871).
    """
    def __init__(self, output_size: int):
        super().__init__()

        self.num_features = output_size
        self.register_parameter(name='gamma',
                                param=nn.Parameter(torch.ones(output_size)))
        self.register_parameter(name='beta',
                                param=nn.Parameter(torch.zeros(output_size)))

    def forward(self, x):
        """Forward pass (modulation)

        :param x: Input features (N, C, *) where * is any number of dims
        :param gamma: (C,)
        :param beta: (C,)
        :return: Output, modulated features (N, C, *), same shape as input
        """
        gamma = self.gamma.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        beta = self.beta.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x)
        return (gamma * x) + beta

    def extra_repr(self) -> str:
        return f'{self.num_features}'


class FiLMedResNetBeforeBlock(ResNetClassifier):
    """ResNet with added FiLM layers between ResNet blocks.

    ResNet with FiLM layers introduced before each new block in the ResNet
    architecture (before the convolutional layer).

    :param film_layers: List of numbers corresponding to blocks within the
        ResNet that FiLM layers should be introduced.
    :param hidden_size: Dimension of final linear layer
    :param resnet: ResNet architecture
    """
    def __init__(self, film_layers: List[int], hidden_size: int = 1024,
                 resnet: str = 'resnet50'):
        super().__init__(hidden_size, resnet)

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

    def forward(self, x):

        # Initial ResNet layers
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        # ResNet body
        for layer in self.resnet_body:
            x = layer(x)

        # ResNet classification layers
        x = self.resnet.avgpool(x)
        x = x.squeeze()
        x = self.leaky_relu(self.resnet.fc(x))
        x = self.linear(x)

        return x


class FiLMedResNetReplaceBN(ResNetClassifier):
    """ResNet with FiLM layers replacing batch normalization layers.

    ResNet with FiLM layers introduced as replacements to some of the ResNet's
    batch normalization layers.

    :param block_replace: List of custom blocks' batch norm layers to replace
    :param bn_replace: List of individual batch norm layers to replace within a
        block
    :param hidden_size: Dimension of final linear layer
    :param resnet: ResNet architecture
    :param replace_downsample: Boolean flag indicating whether to replace the
        batch norm layer in the downsampling step
    :param final_bottleneck_only: Boolean flag indicating whether to replace
        layers specified by `blocko_replace` in every bottlenec, unit, or only
        in the final bottleneck unit. Setting this flag to `True` will only
        replace batch norm layers in the final bottleneck unit in the ResNet
        block if a ResNet block is specified by `block_replace`.
    """
    def __init__(self, block_replace: List[int], bn_replace: List[int],
                 hidden_size: int = 1024, resnet: str = 'resnet50',
                 replace_downsample: bool = False,
                 final_bottleneck_only: bool = False):
        super().__init__(hidden_size, resnet)

        # Must specify blocks and batch norm layers to replace
        assert bn_replace is not None
        if set(bn_replace) != {0}:
            assert block_replace is not None

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
                        if check_if_replace_bn(
                                layer_num, bottleneck_num, bn_num,
                                block_replace, bn_replace, resnet,
                                final_bottleneck_only):
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


def check_if_replace_bn(layer_num: str, bottleneck_num: str, bn_num: str,
                        block_replace: List[int], bn_replace: List[int],
                        resnet: str, final_bottleneck_only: bool = False):
    """Checks whether the passed batch norm layer meets replacement conditions.

    :param layer_num: resnet.{layer_num}
    :param bottleneck_num: resnet.{layer_num}.{bottleneck_num}
    :param bn_num: resnet.{layer_num}.{bottleneck_num}.{bn_num}
    :param block_replace: List of custom blocks' batch norm layers to replace
    :param bn_replace: List of individual batch norm layers to replace within a
        block
    :param resnet: ResNet architecture
    :param final_bottleneck_only: Boolean flag specifying whether to replace
        only the final bottleneck unit layers.
    """
    block_sizes = RESNET_BLOCK_SIZES[resnet]
    block = int(layer_num[-1])
    bn = int(bn_num[-1])
    if block in block_replace and bn in bn_replace:
        if not final_bottleneck_only:
            return True
        elif int(bottleneck_num) == block_sizes[block - 1] - 1:
            return True
        else:
            return False


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
