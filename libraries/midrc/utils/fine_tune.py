"""Helper functions for fine-tuning a subset of model parameters."""
import logging
from typing import List

import torch
from utils.constants import RESNET_BLOCK_SIZES


def get_modules_from_aliases(aliases: List[str], resnet: str,
                             final_bottleneck_only: bool = False):
    """Returns a list of module names from input alias.

    :param aliases: List of string aliases for a subset of ResNet modules to
        fine-tune.
    :param resnet: ResNet architecture
    :param final_bottleneck_only: Boolean flag dictating whether to include
        modules specified by `aliases` in every bottleneck unit, or only in the
        final bottleneck unit. Setting this flag to `True` will only return
        ResNet modules for the final bottleneck unit in the ResNet block if a
        ResNet block is specified by `aliases` (i.e. "block{block_num}").
    """
    block_sizes = RESNET_BLOCK_SIZES[resnet]
    modules = []
    for alias in aliases:
        if alias == 'initial':
            modules.extend(['resnet.conv1', 'resnet.bn1'])
        if 'block' in alias:
            split = alias.split('-')
            block = int(split[0][-1])
            # Add entire block for fine-tuning
            if len(split) == 1 and not final_bottleneck_only:
                modules.append(f'resnet.layer{block}')
            # Add final bottleneck of block for fine-tuning
            elif len(split) == 1 and final_bottleneck_only:
                unit = block_sizes[block - 1] - 1
                modules.append(f'resnet.layer{block}.{unit}')
            # Add specific batch norm layers within block for fine-tuning
            elif len(split) > 1 and not final_bottleneck_only:
                batch_norm = split[1].strip('bn')
                for i in range(block_sizes[block - 1]):
                    for bn in batch_norm:
                        modules.append(f'resnet.layer{block}.{i}.bn{bn}')
            # Add specific batch norm layer in last bottleneck for fine-tuning
            else:
                batch_norm = split[1].strip('bn')
                unit = block_sizes[block - 1] - 1
                for bn in batch_norm:
                    modules.append(f'resnet.layer{block}.{unit}.bn{bn}')
        if alias == 'clf':
            modules.extend(['resnet.fc', 'linear'])
        if alias == 'all':
            modules = [
                'resnet.conv1', 'resnet.bn1', 'resnet.layer1', 'resnet.layer2',
                'resnet.layer3', 'resnet.layer4', 'resnet.fc', 'linear'
            ]
    return modules


def freeze_params(model: torch.nn.Module, fine_tune_modules: List[str]):
    """Freezes ResNet parameters except for those for fine-tuning.

    :param model: PyTorch module to fine-tune
    :param fine_tune_modules: List of ResNet modules to fine-tune

    Common choices for fine_tune_modules include:
        resnet.conv1
        resnet.bn1
        resnet.layer1
        resnet.layer2
        resnet.layer3
        resnet.layer4
        resnet.fc
        linear
    """
    # Freeze all parameters
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Unfreeze module parameters for fine-tuning
    for name, module in model.named_modules():
        if name in fine_tune_modules:
            for param in module.parameters():
                param.requires_grad = True

    logging.debug('Fine-tuning the following parameters...')
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.debug('\t%s', name)
