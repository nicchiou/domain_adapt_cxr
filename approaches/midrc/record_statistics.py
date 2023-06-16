"""Saves gradient and parameter norm statistics."""
from __future__ import division, print_function

import argparse
import json
import logging
import os
import time
from collections import OrderedDict
from typing import Dict

import torch
from models.film import FiLMedResNetReplaceBN
from models.resnet import ResNetClassifier
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import constants
from utils.dataset import MIDRCDataset
from utils.utils import deterministic


def record_stats(model: torch.nn.Module, criterion: torch.nn.Module,
                 dataloaders: Dict[str, DataLoader], device: torch.device):
    """Records statistics for the given module on the provided data set.

    :param args: training arguments
    :param model: model to train
    :param dataloaders: train/valid/test dataloaders
    :param device: device to train model on
    :return: statistics
    :rtype: dict
    """
    trial_results = {}

    # Initialize default settings
    trial_results['train_grad_norm'] = {}
    trial_results['valid_grad_norm'] = {}
    trial_results['test_grad_norm'] = {}

    trial_results['param_fro_norm'] = {}
    trial_results['param_l2_norm'] = {}

    for phase in ['train', 'valid', 'test']:
        model.eval()
        total = 0.0

        # Iterate over dataloader
        for inputs, labels in dataloaders[phase]:

            total += labels.size(0)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            model.zero_grad()

            # Forward pass
            outputs = model(inputs)
            labels = labels.type_as(outputs)

            # Backward pass
            loss = criterion(outputs, labels)
            loss.backward()

            # Calculate gradient norms
            for name, param in model.named_parameters():
                if (param.grad is not None) and (param.requires_grad):
                    cpu_grad = param.grad.data.cpu()
                    if name not in trial_results[f'{phase}_grad_norm'].keys():
                        trial_results[f'{phase}_grad_norm'][name] = 0.
                    trial_results[f'{phase}_grad_norm'][name] += (
                        torch.linalg.norm(cpu_grad).item())

        for name, norm in trial_results[f'{phase}_grad_norm'].items():
            trial_results[f'{phase}_grad_norm'][name] = norm / total

    # Calculate parameter norms
    for name, param in model.named_parameters():
        trial_results['param_fro_norm'][name] = \
            torch.linalg.norm(param.data).item()
        trial_results['param_l2_norm'][name] = \
            torch.linalg.norm(param.data).item()

    return trial_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--res_dir', type=str, required=True)
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--approach', type=str, required=True)
    parser.add_argument('--gpus', type=str, nargs='+', required=True)

    # Dataset
    parser.add_argument('--state', type=str, default='IL',
                        choices=['CA', 'IL', 'IN', 'TX'])
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Total number of samples used for training. '
                        'Setting this flag to -1 uses all available samples '
                        'for each split.')
    parser.add_argument('--valid_fraction', type=float, default=0.1,
                        help='Proportion of training samples used for '
                        'validation.')
    parser.add_argument('--domain', type=str, default='source',
                        choices=['source', 'target'])
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)

    # Model architecture
    parser.add_argument('--resnet', type=str, default='resnet152',
                        choices=['resnet18', 'resnet34', 'resnet50',
                                 'resnet101', 'resnet152'],
                        help='ResNet architecture to use.')
    parser.add_argument('--hidden_size', type=int, default=1024,
                        help='Dimension of the classification linear layer '
                        'after ResNet.')
    parser.add_argument('--film', action='store_true', default=False)
    parser.add_argument('--block_replace', type=int, nargs='+', default=None,
                        choices=[1, 2, 3, 4],
                        help='ResNet blocks within which batch norm layers are '
                        'replaced with FiLM layers.')
    parser.add_argument('--bn_replace', type=int, nargs='+', default=None,
                        choices=[0, 1, 2, 3],
                        help='Numbered batch norm layers to replace with FiLM '
                        'layers. Choices 1, 2, and 3 are within bottleneck '
                        'blocks. Choice 0 represents the batch norm layer '
                        'before any of the ResNet blocks.')
    parser.add_argument('--final_bottleneck_replace', action='store_true',
                        default=False,
                        help='Replace only the batch norm layers in the final '
                        'bottleneck unit of each specified block in '
                        '`block_replace`.')
    parser.add_argument('--replace_downsample', action='store_true',
                        default=False,
                        help='Replaces the batch norm layer in the downsampling'
                        ' step of the ResNet.')
    parser.add_argument('--load_pretrained', type=str, default=None,
                        help='Load a pre-trained model from the given path.')

    # System
    parser.add_argument('--verbose', action='store_true', default=False)

    FLAGS = parser.parse_args()
    timestamp = time.strftime('%Y-%m-%d-%H%M')
    exp_dir = os.path.join(constants.RESULTS_DIR, FLAGS.res_dir, FLAGS.exp_dir)

    # Set up logging
    os.makedirs(os.path.join(FLAGS.log_dir, FLAGS.res_dir), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(FLAGS.log_dir, FLAGS.res_dir,
                              f'{timestamp}_{FLAGS.exp_dir}_{FLAGS.seed}.log'),
        filemode='w',
        format='%(asctime)s\t %(levelname)s:\t%(message)s',
        level=logging.DEBUG,
    )

    # Store statistics in the same exp_dir
    if not os.path.isdir(exp_dir):
        logging.error('Specified experiment directory does not exist.')
        raise OSError

    # Set up torch devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(FLAGS.gpus)
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Running on device cuda:%s',','.join(FLAGS.gpus))

    # Data transforms
    transform = {
        'train':
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True),
            ]),
        'valid':
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True),
            ]),
        'test':
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True),
            ])
    }

    # Number of samples for each split
    n_samples = {
        'train': None if FLAGS.n_samples == -1 \
            else FLAGS.n_samples * (1 - FLAGS.valid_fraction),
        'valid': None if FLAGS.n_samples == -1 \
            else FLAGS.n_samples * FLAGS.valid_fraction,
        'test': None
    }

    # Define PyTorch DataLoaders
    deterministic(FLAGS.seed)
    dataloader_dict = {}
    for split in ['train', 'valid', 'test']:
        dataset = MIDRCDataset(
            os.path.join(constants.METADATA_DIR,
                         f'MIDRC_table_{FLAGS.state}_{split}.csv'),
            n_samples[split],
            transform=transform[split])
        is_train = bool(split == 'train')
        dataloader_dict[split] = DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=is_train,
            drop_last=(is_train and FLAGS.n_samples > FLAGS.batch_size))
        logging.info('Num. state %s samples: %d', split, len(dataset))

    # Define classifier, criterion, and optimizer
    if FLAGS.film:
        net = FiLMedResNetReplaceBN(
            block_replace=FLAGS.block_replace,
            bn_replace=FLAGS.bn_replace,
            hidden_size=FLAGS.hidden_size,
            resnet=FLAGS.resnet,
            replace_downsample=FLAGS.replace_downsample,
            final_bottleneck_only=FLAGS.final_bottleneck_replace
        )
    else:
        net = ResNetClassifier(
            hidden_size=FLAGS.hidden_size,
            resnet=FLAGS.resnet,
        )
    # Load source-trained model and fine-tune a subset of parameters
    state_dict = torch.load(FLAGS.load_pretrained,
                            map_location=torch_device)
    # Modify state_dict to remove DataParallel module wrapper
    new_state_dict = OrderedDict(
        [(key.split('module.')[-1], state_dict[key])
            for key in state_dict])
    net.load_state_dict(new_state_dict)
    logging.info(net)
    if torch.cuda.device_count() > 1:
        gpu_ids = list(range(torch.cuda.device_count()))
        net = torch.nn.DataParallel(net,
                                    device_ids=gpu_ids,
                                    output_device=gpu_ids[-1])
    net.to(torch_device)
    torch_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')

    # ========== EVALUATE ==========
    deterministic(FLAGS.seed)
    results = record_stats(net, torch_criterion, dataloader_dict,
                           torch_device)

    # Write results to JSON files
    n_samples = FLAGS.n_samples if FLAGS.n_samples != -1 else 'all'
    fname = f'{FLAGS.domain}_stats_{FLAGS.state}_n-{n_samples}_{FLAGS.seed}.json'  # pylint: disable=line-too-long
    with open(os.path.join(exp_dir, fname), 'wt', encoding='utf-8') as f:
        json.dump(results, f, indent=4)
