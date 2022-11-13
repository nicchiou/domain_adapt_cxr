"""Helper functions for fine-tuning a subset of model parameters."""
from typing import List
import logging
import torch


def freeze_params(model: torch.nn.Module, fine_tune_modules: List[str]):
    """Freezes ResNet parameters except for those for fine-tuning.

    :param model: model to fine-tune
    :param fine_tune_modules: list of ResNet modules to fine-tune

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
