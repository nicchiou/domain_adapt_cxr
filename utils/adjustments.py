import argparse

import torch


def freeze_layers(args: argparse.Namespace, model: torch.nn.Module):
    """
    Freezes feature-extracting layers of the module in place.

    :param model: model to be frozen
    :type model: torch.nn.Module
    :param fine_tune_layers: fine-tuned layers, options: resnet_fc, linear
    :type fine_tune_layers: list
    """
    for name, param in model.named_parameters():

        # Do not freeze FiLM layer parameters
        if ('gamma' in name or 'beta' in name):
            param.requires_grad = True
        elif args.film and 'film' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

        # Allow fine-tuning of classification layers
        if 'resnet_fc' in args.fine_tune_layers:
            model.resnet.fc.weight.requires_grad = True
            model.resnet.fc.bias.requires_grad = True
        if 'linear' in args.fine_tune_layers:
            model.linear.weight.requires_grad = True
            model.linear.bias.requires_grad = True

        # Allow fine-tuning of batch norm layers
        if 'bn_sparse' in args.fine_tune_layers:
            if 'resnet.layer1.2.bn3' in name:
                param.requires_grad = True
            if 'resnet.layer2.7.bn3' in name:
                param.requires_grad = True
            if 'resnet.layer3.35.bn3' in name:
                param.requires_grad = True
        elif 'bn_all' in args.fine_tune_layers:
            if 'bn3' in name:
                param.requires_grad = True

    print('Fine-tuning the following parameters...', flush=True)
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f'\t{name}', flush=True)
    print(flush=True)


def remove_dropout_layers(model: torch.nn.Module):
    """
    Removes dropout layers from the input model (if present). Modifies the
    model in place.

    :param model: model to be manipulated
    :type model: torch.nn.Module
    """
    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Dropout):
            setattr(model, name, torch.nn.Identity())
