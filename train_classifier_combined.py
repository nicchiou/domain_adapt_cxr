from __future__ import division, print_function

import argparse
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

from models import ResNetClassifier


def deterministic(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_subset_indices(dataset: torch.utils.data.Dataset, n_train_samples: int,
                       random_seed: int):
    """
    Randomly selects n_train_samples number of indices from the dataset,
    stratified by dataset target labels.

    :param dataset: PyTorch train dataset
    :type dataset: torch.utils.data.Dataset
    :param n_train_samples: number of training samples to include in the
        training dataset
    :type n_train_samples: int
    :param random_seed: user-specified random seed for the selection of
        training examples
    :type random_seed: int
    :return: a (n_train_samples,) np.array containing the randomly-selected
        training indices
    """
    if len(dataset.targets) == n_train_samples:
        idx = np.arange(len(dataset))
        np.random.seed(random_seed)
        np.random.shuffle(idx)
    else:
        test_size = len(dataset.targets) - n_train_samples
        idx, _ = train_test_split(list(range(len(dataset.targets))),
                                  test_size=test_size,
                                  stratify=dataset.targets,
                                  random_state=random_seed, shuffle=True)
    return idx


def train(args: argparse.Namespace, model: torch.nn.Module,
          criterion: torch.nn.Module, optimizer: torch.optim,
          scheduler: torch.optim.lr_scheduler, dataloaders: dict, device: str):
    """
    Trains classifier module on the provided data set.

    :param args: training arguments
    :type args: argparse.Namespace
    :param model: model to train
    :type model: torch.nn.Module
    :param criterion: criterion used to train model
    :type criterion: torch.nn.Module
    :param optimizer: optimizer used to train model
    :type optimizer: torch.optim
    :param scheduler: learning rate scheduler
    :type scheduler: torch.optim.lr_scheduler
    :param dataloaders: train and valid loaders
    :type dataloaders: dict
    :param device: device to train model on
    :type device: str
    :return: trained nn.Module and training statistics/metrics
    :rtype: Tuple[nn.Module, dict]
    """

    trial_results = dict()
    trial_results['train_loss'] = list()
    trial_results['train_acc'] = list()
    trial_results['valid_loss'] = list()
    trial_results['valid_acc'] = list()
    start_time = time.time()

    best_model = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(args.num_epochs):
        print('Epoch {}/{}'.format(epoch, args.num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluation mode

            running_loss = 0.0
            running_corrects = 0
            y_true_list = list()
            y_pred_list = list()

            # Iterate over dataloader
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.unsqueeze(1).to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    labels = labels.type_as(outputs)
                    probs = torch.sigmoid(outputs)
                    preds = probs > 0.5
                    loss = criterion(probs, labels)

                    for i in range(len(outputs)):
                        y_true_list.append(labels[i].cpu().data.tolist())
                        y_pred_list.append(probs[i].cpu().data.tolist())

                    # Backward pass
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # Keep track of performance metrics (loss is mean-reduced)
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).item()

            epoch_loss = running_loss / len(y_true_list)
            epoch_acc = float(running_corrects) / len(y_true_list)
            auc = roc_auc_score(y_true_list, y_pred_list)
            trial_results[f'{phase}_loss'].append(epoch_loss)
            trial_results[f'{phase}_acc'].append(epoch_acc)
            trial_results[f'{phase}_auc'] = auc

            if phase == 'valid':
                scheduler.step(epoch_loss)

            print('{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, auc))

            # Save the best model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            elif phase == 'valid' and epoch > 39:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience and args.early_stop:
                    print('\nEarly stopping...\n')
                    break
        else:
            print()
            continue

        break

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best valid acc: {:4f}'.format(best_acc))
    trial_results['best_valid_acc'] = best_acc
    print('Best valid loss: {:4f}'.format(best_loss))
    trial_results['best_valid_loss'] = best_loss
    print()

    # Load best model weights
    model.load_state_dict(best_model)

    return model, trial_results


def test(args: argparse.Namespace, model: torch.nn.Module,
         criterion: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
         device: str):
    """
    Evaluates classifier module on the provided data set.

    :param args: training arguments
    :type args: argparse.Namespace
    :param model: model to train
    :type model: torch.nn.Module
    :param criterion: criterion used to train model
    :type criterion: torch.nn.Module
    :param test_loader: test loader
    :type test_loader: torch.utils.data.DataLoader
    :param device: device to train model on
    :type device: str
    :return: evaluation statistics/metrics
    :rtype: dict
    """

    trial_results = dict()
    model.eval()

    running_loss = 0.0
    running_corrects = 0
    y_true_list = list()
    y_pred_list = list()

    # Iterate over dataloader
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        labels = labels.unsqueeze(1).to(device)

        # Forward pass
        with torch.no_grad():
            outputs = model(inputs)
            labels = labels.type_as(outputs)
            probs = torch.sigmoid(outputs)
            preds = probs > 0.5
            loss = criterion(probs, labels)

            for i in range(len(outputs)):
                y_true_list.append(labels[i].cpu().data.tolist())
                y_pred_list.append(probs[i].cpu().data.tolist())

            # Keep track of performance metrics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()

    test_loss = running_loss / len(y_true_list)
    test_acc = float(running_corrects) / len(y_true_list)
    auc = roc_auc_score(y_true_list, y_pred_list)
    trial_results['test_loss'] = test_loss
    trial_results['test_acc'] = test_acc
    trial_results['test_auc'] = auc

    print('Test Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(
          test_loss, test_acc, auc))
    print()

    return trial_results


real_mimic_train_path = '/shared/rsaas/nschiou2/CXR/data/train/mimic'
real_chexpert_train_path = '/shared/rsaas/nschiou2/CXR/data/train/chexpert'
real_mimic_test_path = '/shared/rsaas/nschiou2/CXR/data/test/mimic'
real_chexpert_test_path = '/shared/rsaas/nschiou2/CXR/data/test/chexpert'


if __name__ == '__main__':

    if torch.cuda.is_available():
        device_num = torch.cuda.current_device()
        device = f'cuda:{device_num}'
    else:
        device = 'cpu'
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_dir', type=str, default='test')
    parser.add_argument('--iter_idx', type=int, default=0)
    parser.add_argument('--num_epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--train_seed', type=int, default=8)
    parser.add_argument('--hidden_size', type=int, default=1024)
    parser.add_argument('--data_sampler_seed', type=int, default=8)
    parser.add_argument('--n_source_samples', type=int, default=10000)
    parser.add_argument('--n_target_samples', type=int, default=10000)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--patience', type=int, default=30)

    args = parser.parse_args()
    timestamp = time.strftime("%Y-%m-%d-%H%M")
    if args.iter_idx != 0:  # If running multiple iters, store in same dir
        exp_dir = os.path.join('experiments', args.exp_dir)
        if not os.path.isdir(exp_dir):
            raise OSError('Specified directory does not exist!')
    else:  # Otherwise, create a new dir
        exp_dir = os.path.join('experiments', f'{args.exp_dir}')
        os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, f'args_{args.iter_idx}.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    deterministic(args.train_seed)

    transform = {
        'train':
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomResizedCrop((224), scale=(0.9, 1)),
            transforms.RandomHorizontalFlip(),
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

    # Load PyTorch datasets from directories
    mimic_train = datasets.ImageFolder(real_mimic_train_path,
                                       transform['train'])
    chexpert_train = datasets.ImageFolder(real_chexpert_train_path,
                                          transform['train'])
    mimic_test = datasets.ImageFolder(real_mimic_test_path,
                                      transform['test'])
    chexpert_test = datasets.ImageFolder(real_chexpert_test_path,
                                         transform['test'])

    # Define classifier, criterion, and optimizer
    model = ResNetClassifier(hidden_size=args.hidden_size)
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=10, threshold=1e-4, min_lr=1e-10,
        verbose=True)

    # ========== TRAINING / EVALUATION PHASE ========== #
    # Select a stratified subset of the training dataset to use
    source_subset_idx = get_subset_indices(chexpert_train,
                                           args.n_source_samples,
                                           args.data_sampler_seed)
    source_subset = torch.utils.data.Subset(chexpert_train, source_subset_idx)
    target_subset_idx = get_subset_indices(mimic_train,
                                           args.n_target_samples,
                                           args.data_sampler_seed)
    target_subset = torch.utils.data.Subset(mimic_train, target_subset_idx)
    subset = torch.utils.data.ConcatDataset([source_subset, target_subset])
    print('train dataset size: ', len(subset))

    # Split into train and validation sets and create PyTorch Dataloaders
    train_dataset, valid_dataset = torch.utils.data.random_split(
        subset,
        [int(np.floor(0.8 * len(subset))), int(np.ceil(0.2 * len(subset)))],
        generator=torch.Generator().manual_seed(args.data_sampler_seed))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True)
    mimic_test_loader = torch.utils.data.DataLoader(
        mimic_test, batch_size=args.batch_size, shuffle=False)
    chexpert_test_loader = torch.utils.data.DataLoader(
        chexpert_test, batch_size=args.batch_size, shuffle=False)
    dataloaders = {'train': train_loader, 'valid': valid_loader}

    model, trial_results = train(args, model, criterion, optimizer,
                                 lr_scheduler, dataloaders, device)
    mimic_test_results = test(args, model, criterion, mimic_test_loader,
                              device)
    chexpert_test_results = test(args, model, criterion,
                                 chexpert_test_loader, device)
    trial_results.update(
        {f'mimic_{k}': v for k, v in mimic_test_results.items()})
    trial_results.update(
        {f'chexpert_{k}': v
            for k, v in chexpert_test_results.items()})
    results_df = pd.DataFrame(columns=list(trial_results.keys()))
    results_df = results_df.append(trial_results, ignore_index=True)
    results_df.to_parquet(
        os.path.join(exp_dir, f'results_{args.iter_idx}.parquet'),
        index=False)
    torch.save(model.state_dict(),
               os.path.join(exp_dir,
                            f'combined_checkpoint_{args.iter_idx}.pt'))
