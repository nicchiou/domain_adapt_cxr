"""Trains a ResNet on MIDRC data."""
from __future__ import division, print_function

import argparse
import copy
import json
import logging
import os
import shutil
import time
from typing import Dict

import torch
from models.film import FiLMedResNetReplaceBN
from models.resnet import ResNetClassifier
from sklearn.metrics import roc_auc_score
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms
from utils import constants
from utils.dataset import MIDRCDataset
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
    trial_results['train_epoch_grad_norm'] = []
    trial_results['valid_epoch_loss'] = []
    trial_results['valid_epoch_acc'] = []
    trial_results['valid_epoch_auc'] = []
    trial_results['valid_epoch_grad_norm'] = []

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
                for p in list(filter(
                        lambda p: p.grad is not None, model.parameters())):
                    running_grad_norm += p.grad.data.norm(2).item()

            # Evaluate training predictions against ground truth labels
            epoch_loss = running_loss / total
            epoch_acc = float(running_corrects) / total
            try:
                epoch_auc = roc_auc_score(y_true, y_pred)
            except ValueError:
                # Ill-defined when y_true only consists of one class
                epoch_auc = 0.0
            grad_norm_avg = running_grad_norm / total
            trial_results[f'{phase}_epoch_loss'].append(epoch_loss)
            trial_results[f'{phase}_epoch_acc'].append(epoch_acc)
            trial_results[f'{phase}_epoch_auc'].append(epoch_auc)
            trial_results[f'{phase}_epoch_grad_norm'].append(grad_norm_avg)

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
    try:
        test_auc = roc_auc_score(y_true, y_pred)
    except ValueError:
        # Ill-defined when y_true only consists of one class
        test_auc = 0.0
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
    parser.add_argument('--states', type=str, nargs='+', required=True,
                        choices=['CA', 'IL', 'IN', 'TX'],
                        help='States to use for combined training dataset and '
                        'evaluating each state\'s test performance.')
    parser.add_argument('--n_samples', type=int, default=-1,
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
    parser.add_argument('--replace_downsample', action='store_true',
                        default=False,
                        help='Replaces the batch norm layer in the downsampling'
                        ' step of the ResNet.')

    # Optimization and model-fitting hyperparameters
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)

    # Learning rate scheduler
    parser.add_argument('--decay_factor', type=float, default=0.3,
                        help='Factor to muptiply the current learning rate by '
                        'when decreasing the learning rate after a plateau.')
    parser.add_argument('--patience', type=int, default=10,
                        help='performance metric does not improve after '
                        '`early_stop` number of epochs.')
    parser.add_argument('--min_lr', type=float, default=1e-8,
                        help='Lower-bound on learning rate decay.')

    # Model selection
    parser.add_argument('--early_stop', type=int, default=-1,
                        help='Stops training early if the optimized '
                        'performance metric does not improve after early_stop '
                        'epochs.')
    parser.add_argument('--early_stopping_metric', type=str, default='auc',
                        choices=['loss', 'acc', 'auc'])
    parser.add_argument('--min_epochs', type=int, default=0,
                        help='Minimum number of epochs to train for before '
                        'early stopping starts.')

    # System
    parser.add_argument('--verbose', action='store_true', default=False)

    FLAGS = parser.parse_args()
    timestamp = time.strftime('%Y-%m-%d-%H%M')
    exp_dir = os.path.join(constants.RESULTS_DIR, FLAGS.res_dir, FLAGS.exp_dir)

    # Basic assertions
    # TODO: support for variable number of samples in the combined dataset
    assert FLAGS.n_samples == -1

    # Set up logging
    os.makedirs(os.path.join(FLAGS.log_dir, FLAGS.res_dir), exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(FLAGS.log_dir, FLAGS.res_dir,
                              f'{timestamp}_{FLAGS.exp_dir}_{FLAGS.seed}.log'),
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
    train_dataloaders = {}
    for split in ['train', 'valid']:
        datasets = {}
        for state in FLAGS.states:
            datasets[state] = MIDRCDataset(
                os.path.join(constants.METADATA_DIR,
                            f'MIDRC_table_{state}_{split}.csv'),
                n_samples[split],
                transform=transform[split])
        dataset = ConcatDataset(list(datasets.values()))
        is_train = bool(split == 'train')
        train_dataloaders[split] = DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=is_train,
            drop_last=(is_train and 2 * FLAGS.n_samples > FLAGS.batch_size))
        logging.info('Num. combined %s samples: %d', split, len(dataset))

    test_dataloaders = {}
    for state in FLAGS.states:
        dataset = MIDRCDataset(
            os.path.join(constants.METADATA_DIR,
                        f'MIDRC_table_{state}_test.csv'),
            n_samples['test'],
            transform=transform['test'])
        test_dataloaders[state] = DataLoader(
            dataset, batch_size=FLAGS.batch_size, shuffle=False,
            drop_last=False)
        logging.info('Num. %s test samples: %d', state, len(dataset))

    # Define classifier, criterion, and optimizer
    deterministic(FLAGS.seed)
    if FLAGS.film:
        net = FiLMedResNetReplaceBN(block_replace=FLAGS.block_replace,
                                    bn_replace=FLAGS.bn_replace,
                                    hidden_size=FLAGS.hidden_size,
                                    resnet=FLAGS.resnet,
                                    replace_downsample=FLAGS.replace_downsample)
    else:
        net = ResNetClassifier(hidden_size=FLAGS.hidden_size,
                               resnet=FLAGS.resnet)
    logging.info(net)
    if torch.cuda.device_count() > 1:
        net = torch.nn.DataParallel(net)
    net.to(torch_device)

    # Instantiate criterion, optimizer, and learning rate scheduler
    torch_criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    torch_optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, net.parameters()), lr=FLAGS.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        torch_optimizer, factor=FLAGS.decay_factor, patience=FLAGS.patience,
        threshold=1e-4, min_lr=FLAGS.min_lr, verbose=FLAGS.verbose)

    # Train and evaluate model
    train_results, net = train(FLAGS, net, torch_criterion, torch_optimizer,
                               lr_scheduler, train_dataloaders, torch_device)
    test_results = {}
    for state in FLAGS.states:
        test_results.update(
            {f'{state}_{k}': v for k, v in test(
                FLAGS, net, torch_criterion, test_dataloaders[state],
                torch_device).items()})
        logging.info('%s Test Loss: %.4f', state,
                     test_results[f'{state}_test_loss'])
        logging.info('%s Test Accuracy: %.4f', state,
                     test_results[f'{state}_test_acc'])
        logging.info('%s Test ROC AUC: %.4f', state,
                     test_results[f'{state}_test_auc'])

    # Write results to JSON files
    with open(os.path.join(exp_dir,
                           f'{FLAGS.domain}_train_{FLAGS.seed}.json'),
              'wt', encoding='utf-8') as f:
        json.dump(train_results, f, indent=4)
    results = {
        'approach': FLAGS.approach,
        'domain': FLAGS.domain,
        'states': FLAGS.states,
        'seed': FLAGS.seed,
        'n_samples': FLAGS.n_samples,
        'early_stopping_metric': FLAGS.early_stopping_metric,
        f'{FLAGS.states[0]}_loss': test_results[f'{FLAGS.states[0]}_test_loss'],
        f'{FLAGS.states[0]}_acc': test_results[f'{FLAGS.states[0]}_test_acc'],
        f'{FLAGS.states[0]}_auc': test_results[f'{FLAGS.states[0]}_test_auc'],
        f'{FLAGS.states[1]}_loss': test_results[f'{FLAGS.states[1]}_test_loss'],
        f'{FLAGS.states[1]}_acc': test_results[f'{FLAGS.states[1]}_test_acc'],
        f'{FLAGS.states[1]}_auc': test_results[f'{FLAGS.states[1]}_test_auc'],
    }
    with open(os.path.join(exp_dir, f'results_{FLAGS.seed}.json'),
              'wt', encoding='utf-8') as f:
        json.dump(results, f, indent=4)

    # Save model checkpoint
    torch.save(net.state_dict(),
               os.path.join(exp_dir,
                            f'{FLAGS.domain}_checkpoint_{FLAGS.seed}.pt'))

    # Copy .log file
    src = os.path.join(FLAGS.log_dir, FLAGS.res_dir,
                       f'{timestamp}_{FLAGS.exp_dir}_{FLAGS.seed}.log')
    des = os.path.join(exp_dir, f'train_log_{FLAGS.seed}.log')
    shutil.copyfile(src, des)
