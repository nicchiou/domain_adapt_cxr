"""Trains a ResNet on MIDRC data."""
from __future__ import division, print_function

import argparse
import copy
import json
import logging
import os
import time
from collections import OrderedDict
from typing import Dict

import torch
from models.film import FiLMedResNetReplaceBN
from models.resnet import ResNetClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader
from torchvision import transforms
from utils import constants
from utils.dataset import MIDRCDataset
from utils.fine_tune import get_modules_from_aliases, freeze_params
from utils.utils import deterministic


def train(args: argparse.Namespace, model: torch.nn.Module,
          criterion: torch.nn.Module, optimizer: torch.optim,
          scheduler: torch.optim.lr_scheduler,
          dataloaders: Dict[str, DataLoader], device: torch.device):
    """Trains the classifier module on the provided data set.

    :param args: training arguments
    :param model: model to train
    :param criterion: criterion to optimize
    :param optimizer: optimizer used to train model
    :param scheduler: learning rate scheduler
    :param dataloaders: train/valid/test dataloaders
    :param device: device to train model on
    :return: trained nn.Module and training statistics/metrics
    :rtype: Tuple[dict, nn.Module]
    """
    start_time = time.time()

    # Initialize training state and default settings
    epochs_without_improvement = 0
    trial_results = {}
    trial_results['train_epoch_loss'] = []
    trial_results['train_epoch_acc'] = []
    trial_results['train_epoch_auc'] = []
    trial_results['valid_epoch_loss'] = []
    trial_results['valid_epoch_acc'] = []
    trial_results['valid_epoch_auc'] = []
    trial_results['grad_norm'] = []

    best_model = copy.deepcopy(model.state_dict())
    best_model_epoch = 0
    best_valid_metrics = {}
    best_valid_metrics['loss'] = 1e6
    best_valid_metrics['acc'] = -1
    best_valid_metrics['auc'] = -1
    best_valid_metric = 1e6 if args.early_stopping_metric == 'loss' else -1

    for epoch in range(1, args.epochs + 1):

        if args.verbose:
            print(f'Epoch {epoch}/{args.epochs}', flush=True)
            print('-' * 10, flush=True)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to validation mode

            running_loss = 0.0
            running_corrects = 0.0
            running_grad_norm = 0.0
            total = 0.0
            y_prob = []
            y_pred = []
            y_true = []

            # Iterate over dataloader
            for inputs, labels in dataloaders[phase]:

                total += labels.size(0)
                inputs = inputs.to(device)
                labels = labels.to(device)

                # Zero the parameter gradients
                model.zero_grad()
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    labels = labels.type_as(outputs)
                    probs = torch.sigmoid(outputs)
                    preds = probs > 0.5

                    y_prob.extend(probs.cpu().data.tolist())
                    y_pred.extend(preds.cpu().data.tolist())
                    y_true.extend(labels.cpu().data.tolist())

                    # Backward pass
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Keep track of performance metrics (loss is mean-reduced)
                running_loss += loss.item() * labels.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

                # Calculate gradient norms
                if phase == 'train':
                    for p in list(filter(
                            lambda p: p.grad is not None, model.parameters())):
                        running_grad_norm += p.grad.data.norm(2).item()

            # Evaluate training predictions against ground truth labels
            epoch_loss = running_loss / total
            epoch_acc = float(running_corrects) / total
            epoch_auc = roc_auc_score(y_true, y_pred)
            trial_results[f'{phase}_epoch_loss'].append(epoch_loss)
            trial_results[f'{phase}_epoch_acc'].append(epoch_acc)
            trial_results[f'{phase}_epoch_auc'].append(epoch_auc)
            if phase == 'train':
                grad_norm_avg = running_grad_norm / total
                trial_results['grad_norm'].append(grad_norm_avg)

            # Update LR scheduler with current validation loss
            if phase == 'valid':
                scheduler.step(epoch_loss)

            # Print metrics to stdout
            if args.verbose:
                print(f'{phase.capitalize()} Loss: {epoch_loss:.4f}\t '
                      f'Accuracy: {epoch_acc:.4f}\t '
                      f'AUC: {epoch_auc:.4f}', flush=True)

            # Save the best model at each epoch, using validation metric
            eps = 0.001
            if phase == 'valid':
                check_loss = (args.early_stopping_metric == 'loss' and
                              epoch_loss < best_valid_metric and
                              best_valid_metric - epoch_loss >= eps)
                check_acc = (args.early_stopping_metric == 'acc' and
                             epoch_acc > best_valid_metric and
                             epoch_acc - best_valid_metric >= eps)
                check_auc = (args.early_stopping_metric == 'auc' and
                             epoch_auc > best_valid_metric and
                             epoch_auc - best_valid_metric >= eps)
                if check_loss or check_acc or check_auc:
                    # Reset early stopping epochs w/o improvement
                    epochs_without_improvement = 0
                    # Record best validation metrics
                    best_valid_metrics['loss'] = epoch_loss
                    best_valid_metrics['acc'] = epoch_acc
                    best_valid_metrics['auc'] = epoch_auc
                    best_valid_metric = \
                        best_valid_metrics[args.early_stopping_metric]
                    # Save best model as a deepcopy
                    best_model = copy.deepcopy(model.state_dict())
                    best_model_epoch = epoch
                elif epoch > args.min_epochs:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= args.early_stop and \
                            args.early_stop != -1:
                        trial_results['epoch_early_stop'] = epoch
                        if args.verbose:
                            print('\nEarly stopping...\n', flush=True)
                        break
        # This block only executes if the for loop is not exited with a break
        # statement (early stopping)
        else:
            if args.verbose:
                print(flush=True)
            continue
        # Inner loop break forces outer loop break
        break

    time_elapsed = time.time() - start_time

    # ======== SUMMARY ======== #
    train_loss = trial_results['train_epoch_loss'][best_model_epoch - 1]
    train_acc = trial_results['train_epoch_acc'][best_model_epoch - 1]
    train_auc = trial_results['train_epoch_auc'][best_model_epoch - 1]
    valid_loss = trial_results['valid_epoch_loss'][best_model_epoch - 1]
    valid_acc = trial_results['valid_epoch_acc'][best_model_epoch - 1]
    valid_auc = trial_results['valid_epoch_auc'][best_model_epoch - 1]
    assert valid_loss == best_valid_metrics['loss']
    assert valid_acc == best_valid_metrics['acc']
    assert valid_auc == best_valid_metrics['auc']

    if args.verbose:
        print(f'Training complete in {time_elapsed // 60:.0f}m '
              f'{time_elapsed % 60:.0f}s', flush=True)
        print(f'Best valid loss: {valid_loss:4f}', flush=True)
        print(f'Best valid accuracy: {valid_acc:4f}', flush=True)
        print(f'Best valid AUC ROC: {valid_auc:.4f}', flush=True)
        print(flush=True)
    logging.info('Training complete in %.0fm %.0fs', time_elapsed // 60,
                 time_elapsed % 60)
    logging.info('Train Loss: %.4f', train_loss)
    logging.info('Train Accuracy: %.4f', train_acc)
    logging.info('Train ROC AUC: %.4f', train_auc)
    logging.info('Valid Loss: %.4f', valid_loss)
    logging.info('Valid Accuracy: %.4f', valid_acc)
    logging.info('Valid ROC AUC: %.4f', valid_auc)

    trial_results['epoch_early_stop'] = best_model_epoch
    trial_results['train_loss'] = train_loss
    trial_results['train_acc'] = train_acc
    trial_results['train_auc'] = train_auc
    trial_results['valid_loss'] = valid_loss
    trial_results['valid_acc'] = valid_acc
    trial_results['valid_auc'] = valid_auc

    # Load best model weights
    model.load_state_dict(best_model)

    return trial_results, model


def test(args: argparse.Namespace, model: torch.nn.Module,
         criterion: torch.nn.Module, test_loader: DataLoader,
         device: torch.device):
    """Evaluates the classifier module on the provided held-out test data.

    :param args: training arguments
    :param model: model to evaluate
    :param criterion: criterion to optimize
    :param test_loader: test loader
    :param device: device to train model on
    :return: evaluation statistics/metrics
    :rtype: dict
    """
    trial_results = {}
    model.eval()

    running_loss = 0.0
    running_corrects = 0.0
    total = 0.0
    y_prob = []
    y_pred = []
    y_true = []

    # Iterate over dataloader
    for inputs, labels in test_loader:

        total += labels.size(0)
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            labels = labels.type_as(outputs)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5

            y_prob.extend(probs.cpu().data.tolist())
            y_pred.extend(preds.cpu().data.tolist())
            y_true.extend(labels.cpu().data.tolist())

            loss = criterion(outputs, labels)

            # Keep track of performance metrics
            running_loss += loss.item() * labels.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    test_loss = running_loss / total
    test_acc = float(running_corrects) / total
    test_auc = roc_auc_score(y_true, y_pred)
    trial_results['test_loss'] = test_loss
    trial_results['test_acc'] = test_acc
    trial_results['test_auc'] = test_auc

    if args.verbose:
        print(f'Test Loss: {test_loss:.4f}\t '
              f'Accuracy: {test_acc:.4f}\t AUC: {test_auc:.4f}\n', flush=True)

    return trial_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--res_dir', type=str, required=True)
    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, required=True)
    parser.add_argument('--approach', type=str, required=True)
    parser.add_argument('--gpus', type=str, nargs='+', required=True)

    # Dataset
    parser.add_argument('--train_state', type=str, default='IL',
                        choices=['CA', 'IL', 'IN', 'NC', 'TX'])
    parser.add_argument('--test_state', type=str, default='CA',
                        choices=['CA', 'IL', 'IN', 'NC', 'TX'])
    parser.add_argument('--n_samples', type=int, default=10000,
                        help='Total number of samples used for training. '
                        'Setting this flag to -1 uses all available samples '
                        'for each split.')
    parser.add_argument('--valid_fraction', type=float, default=0.1,
                        help='Proportion of training samples used for '
                        'validation.')
    parser.add_argument('--domain', type=str, default='source',
                        choices=['source', 'target'],
                        help='Specifies whether to train a new model (source) '
                        'or fine-tune an existing model (target).')

    # Model architecture
    parser.add_argument('--resnet', type=str, default='resnet50',
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

    # Fine-tuning and/or inference
    parser.add_argument('--load_pretrained', type=str, default=None,
                        help='Load a pre-trained model from the given path.')
    parser.add_argument('--fine_tune_modules', type=str, nargs='+',
                        default=None,
                        help='List of aliased ResNet modules to fine-tune.')
    parser.add_argument('--final_bottleneck_fine_tune', action='store_true',
                        default=False,
                        help='Fine-tune only the batch norm layers in the '
                        'final bottleneck unit of each specified block by '
                        '`fine_tune_modules`.')
    parser.add_argument('--inference_only', action='store_true', default=False)

    # Optimization and model-fitting hyperparameters
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)

    # Learning rate scheduler
    parser.add_argument('--decay_factor', type=float, default=0.3,
                        help='Factor to muptiply the current learning rate by '
                        'when decreasing the learning rate after a plateau.')
    parser.add_argument('--patience', type=int, default=10,
                        help='Number of epochs to wait before decaying '
                        'the learning rate.')
    parser.add_argument('--min_lr', type=float, default=1e-8,
                        help='Lower-bound on learning rate decay.')

    # Model selection
    parser.add_argument('--early_stop', type=int, default=-1,
                        help='Stops training early if the optimized '
                        'performance metric does not improve after '
                        '`early_stop` number of epochs.')
    parser.add_argument('--early_stopping_metric', type=str, default='auc',
                        choices=['loss', 'acc', 'auc'])
    parser.add_argument('--min_epochs', type=int, default=0,
                        help='Minimum number of epochs to train for before '
                        'early stopping starts.')

    # System
    parser.add_argument('--verbose', action='store_true', default=False)

    FLAGS = parser.parse_args()
    timestamp = time.strftime('%Y-%m-d-%H%M')
    exp_dir = os.path.join(constants.RESULTS_DIR, FLAGS.res_dir, FLAGS.exp_dir)

    # Basic assertions
    if FLAGS.load_pretrained is not None:
        if 'film' in FLAGS.load_pretrained:
            assert (FLAGS.film and FLAGS.block_replace is not None and
                    FLAGS.bn_replace is not None), (
                        'Must specify FiLM model type.')
    if FLAGS.fine_tune_modules is not None:
        if 'all' in FLAGS.fine_tune_modules:
            assert len(FLAGS.fine_tune_modules) == 1, (
                'Already fine-tuning all parameters.')
        for alias in FLAGS.fine_tune_modules:
            basic_check = alias in ['all', 'initial', 'clf']
            block_check = True
            if 'block' in alias:
                split = alias.split('-')
                block = int(split[0][-1])
                block_check = block in {1, 2, 3, 4}
                if len(split) == 2:
                    batch_norm = split[1].strip('bn')
                    for bn in batch_norm:
                        block_check = block_check and int(bn) in {1, 2, 3}
            assert basic_check or block_check, 'Alias not supported.'

    # Set up logging
    os.makedirs(os.path.join(FLAGS.log_dir, FLAGS.res_dir), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(FLAGS.log_dir, FLAGS.res_dir,
                              f'{FLAGS.exp_dir}_{FLAGS.seed}.log'),
        filemode='w',
        format='%(asctime)s\t %(levelname)s:\t%(message)s',
        level=logging.DEBUG,
    )

    # Store multiple training iterations in the same exp_dir
    if FLAGS.seed != 0:
        if not os.path.isdir(exp_dir):
            logging.error('Specified experiment directory does not exist.')
            raise OSError
    else:
        os.makedirs(exp_dir, exist_ok=True)

    # Save command line arguments
    with open(os.path.join(exp_dir, f'args_{FLAGS.seed}.json'), 'wt',
              encoding='utf-8') as f:
        json.dump(FLAGS.__dict__, f, indent=4)

    # Set up training devices
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(FLAGS.gpus)
    torch_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info('Training on device cuda:%s',','.join(FLAGS.gpus))

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
    train_state_dataloaders = {}
    for split in ['train', 'valid', 'test']:
        dataset = MIDRCDataset(
            os.path.join(constants.METADATA_DIR,
                         f'MIDRC_table_{FLAGS.train_state}_{split}.csv'),
            n_samples[split],
            transform=transform[split])
        is_train = bool(split == 'train')
        train_state_dataloaders[split] = DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=is_train,
            drop_last=(is_train and FLAGS.n_samples > FLAGS.batch_size))
        logging.info('Num. train state %s samples: %d', split, len(dataset))

    dataset = MIDRCDataset(
        os.path.join(constants.METADATA_DIR,
                     f'MIDRC_table_{FLAGS.test_state}_test.csv'),
        n_samples['test'],
        transform=transform['test'])
    test_state_dataloader = DataLoader(
        dataset, batch_size=FLAGS.batch_size, shuffle=False, drop_last=False)
    logging.info('Num. test state test samples: %d', len(dataset))

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
    if FLAGS.load_pretrained is not None:
        state_dict = torch.load(FLAGS.load_pretrained,
                                map_location=torch_device)
        # Modify state_dict to remove DataParallel module wrapper
        new_state_dict = OrderedDict(
            [(key.split('module.')[-1], state_dict[key])
             for key in state_dict])
        net.load_state_dict(new_state_dict)
        if FLAGS.domain == 'target':
            modules = get_modules_from_aliases(
                FLAGS.fine_tune_modules,
                resnet=FLAGS.resnet,
                final_bottleneck_only=FLAGS.final_bottleneck_fine_tune,
            )
            freeze_params(net, modules)
    logging.info(net)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(torch_device)

    # Instantiate criterion, optimizer, and learning rate schedule
    torch_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    torch_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=FLAGS.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        torch_optimizer, factor=FLAGS.decay_factor, patience=FLAGS.patience,
        threshold=1e-4, min_lr=FLAGS.min_lr, verbose=True)

    # ========== TRAIN ==========
    deterministic(FLAGS.seed)
    if not FLAGS.inference_only:
        train_state_results, net = train(FLAGS, net, torch_criterion,
                                         torch_optimizer, lr_scheduler,
                                         train_state_dataloaders, torch_device)
    else:
        # Training results do not exist for pre-trained inference-only models
        train_state_results = {}

    # ========= TEST ==========
    train_state_test_results = test(FLAGS, net, torch_criterion,
                                    train_state_dataloaders['test'],
                                    torch_device)
    train_state_results.update(train_state_test_results)
    logging.info('Train State Loss: %.4f', train_state_results['test_loss'])
    logging.info('Train State Accuracy: %.4f', train_state_results['test_acc'])
    logging.info('Train State ROC AUC: %.4f', train_state_results['test_auc'])
    test_state_results = test(FLAGS, net, torch_criterion,
                              test_state_dataloader, torch_device)
    logging.info('Test State Loss: %.4f', test_state_results['test_loss'])
    logging.info('Test State Accuracy: %.4f', test_state_results['test_acc'])
    logging.info('Test State ROC AUC: %.4f', test_state_results['test_auc'])

    # Write results to JSON files
    with open(os.path.join(exp_dir,
                           f'{FLAGS.domain}_train_{FLAGS.seed}.json'),
              'wt', encoding='utf-8') as f:
        json.dump(train_state_results, f, indent=4)
    results = {
        'approach': FLAGS.approach,
        'domain': FLAGS.domain,
        'train_state': FLAGS.train_state,
        'test_state': FLAGS.test_state,
        'iter': FLAGS.seed,
        'n_samples': FLAGS.n_samples,
        f'{FLAGS.train_state}_loss': train_state_results['test_loss'],
        f'{FLAGS.train_state}_acc': train_state_results['test_acc'],
        f'{FLAGS.train_state}_auc': train_state_results['test_auc'],
        f'{FLAGS.test_state}_loss': test_state_results['test_loss'],
        f'{FLAGS.test_state}_acc': test_state_results['test_acc'],
        f'{FLAGS.test_state}_auc': test_state_results['test_auc'],
    }
    with open(os.path.join(exp_dir, f'results_{FLAGS.seed}.json'),
              'wt', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    # Save model checkpoint
    torch.save(net.state_dict(),
               os.path.join(exp_dir,
                            f'{FLAGS.domain}_checkpoint_{FLAGS.seed}.pt'))
