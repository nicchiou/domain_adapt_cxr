from __future__ import division, print_function

import argparse
import copy
import json
import os
import time
from tqdm import tqdm

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torchvision import datasets, transforms

from models.resnet import ResNetOrig
from utils import constants
from utils.dataset import CXRImageFolder
from utils.data_sampler import get_subset_indices
from utils.utils import deterministic


def train(args: argparse.Namespace, model: torch.nn.Module,
          criterion: torch.nn.Module, optimizer: torch.optim,
          scheduler: torch.optim.lr_scheduler, dataloaders: dict, device: str):
    """
    Trains classifier module on the provided data set.

    :param args: training arguments
    :param model: model to train
    :param criterion: criterion used to train model
    :param optimizer: optimizer used to train model
    :param scheduler: learning rate scheduler
    :param dataloaders: train and valid loaders
    :param device: device to train model on
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
    final_train_loss = 0.0
    final_train_acc = 0.0
    num_epochs = args.num_epochs
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1), flush=True)
        print('-' * 10, flush=True)

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

            # Update LR scheduler with current validation loss
            if phase == 'valid':
                scheduler.step(epoch_loss)

            # Keep track of current training loss and accuracy
            if phase == 'train':
                final_train_loss = epoch_loss
                final_train_acc = epoch_acc

            # Print metrics to stdout
            print('{} Loss: {:.4f} Acc: {:.4f} AUC: {:.4f}'.format(
                phase, epoch_loss, epoch_acc, auc), flush=True)

            # Save the best model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model = copy.deepcopy(model.state_dict())
                epochs_no_improve = 0
            if phase == 'valid' and epoch_loss < best_loss:
                best_loss = epoch_loss
            elif phase == 'valid' and epoch > 39:
                epochs_no_improve += 1
                if epochs_no_improve >= args.patience and args.early_stop:
                    print('\nEarly stopping...\n', flush=True)
                    trial_results['epoch_early_stop'] = epoch + 1
                    break
        else:
            print(flush=True)
            continue

        break

    time_elapsed = time.time() - start_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60), flush=True)
    print('Best valid acc: {:4f}'.format(best_acc), flush=True)
    trial_results['best_valid_acc'] = best_acc
    print('Best valid loss: {:4f}'.format(best_loss), flush=True)
    trial_results['best_valid_loss'] = best_loss
    print(flush=True)

    # Save final model results
    trial_results['final_train_loss'] = final_train_loss
    trial_results['final_valid_loss'] = epoch_loss
    trial_results['final_train_acc'] = final_train_acc
    trial_results['final_valid_acc'] = epoch_acc

    # Save final model weights
    final_model = copy.deepcopy(model)

    # Load best model weights
    model.load_state_dict(best_model)

    return model, final_model, trial_results


def test(args: argparse.Namespace, model: torch.nn.Module,
         criterion: torch.nn.Module, test_loader: torch.utils.data.DataLoader,
         device: str):
    """
    Evaluates classifier module on the provided data set.

    :param args: training arguments
    :param model: model to train
    :param criterion: criterion used to train model
    :param test_loader: test loader
    :param device: device to train model on
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
          test_loss, test_acc, auc), flush=True)
    print(flush=True)

    return trial_results


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_dir', type=str, required=True)
    parser.add_argument('--iter_idx', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='chexpert',
                        choices=['chexpert', 'mimic'])
    parser.add_argument('--resnet', type=str, default='resnet50')
    parser.add_argument('--load_trained_model', action='store_true')
    parser.add_argument('--use_imagenet', action='store_true')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--save_embeddings', action='store_true')
    parser.add_argument('--embed_dataset', type=str, default='chexpert_A',
                        choices=['chexpert_A', 'chexpert_B',
                                 'chexpert_F', 'chexpert_M'])
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--no_drop_last', action='store_false')
    parser.add_argument('--train_seed', type=int, default=8)
    parser.add_argument('--data_sampler_seed', type=int, default=8)
    parser.add_argument('--n_samples', type=int, default=20000)
    parser.add_argument('--valid_fraction', type=float, default=None)
    parser.add_argument('--early_stop', action='store_true')
    parser.add_argument('--patience', type=int, default=30)

    args = parser.parse_args()
    timestamp = time.strftime("%Y-%m-%d-%H%M")
    if args.iter_idx != 0:  # If running multiple iters, store in same dir
        exp_dir = os.path.join('experiments', args.exp_dir)
        if not os.path.isdir(exp_dir):
            raise OSError('Specified directory does not exist!')
    else:  # Otherwise, create a new dir
        exp_dir = os.path.join('experiments', args.exp_dir)
        os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, f'args_{args.iter_idx}.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=4)
    deterministic(args.train_seed)

    if torch.cuda.is_available():
        device_num = torch.cuda.current_device()
        device = f'cuda:{device_num}'
    else:
        device = 'cpu'
    print(f'Training on device: {device}')

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
    if args.dataset == 'chexpert':
        dataset_train = datasets.ImageFolder(
            constants.REAL_CHEXPERT_TRAIN_PATH, transform['train'])
        dataset_test = datasets.ImageFolder(
            constants.REAL_CHEXPERT_TEST_PATH, transform['test'])
    elif args.dataset == 'mimic':
        dataset_train = datasets.ImageFolder(
            constants.REAL_MIMIC_TRAIN_PATH, transform['train'])
        dataset_test = datasets.ImageFolder(
            constants.REAL_MIMIC_TEST_PATH, transform['test'])

    # Define classifier, criterion, and optimizer
    model = ResNetOrig(resnet=args.resnet)
    print(model, flush=True)
    model = model.to(device)
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=10, threshold=1e-4, min_lr=1e-10,
        verbose=True)

    # Control deterministic behavior
    deterministic(args.train_seed)

    # Select a stratified subset of the training dataset to use
    subset_idx = get_subset_indices(dataset_train, args.n_samples,
                                    args.data_sampler_seed)
    subset = torch.utils.data.Subset(dataset_train, subset_idx)

    # Split into train and validation sets and create PyTorch Dataloaders
    train_dataset, valid_dataset = torch.utils.data.random_split(
        subset,
        [int(np.floor(0.8 * len(subset))), int(np.ceil(0.2 * len(subset)))],
        generator=torch.Generator().manual_seed(args.data_sampler_seed))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=args.no_drop_last)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, shuffle=False)
    dataloaders = {'train': train_loader, 'valid': valid_loader}

    # Load existing trained model or train a new one
    if args.load_trained_model:
        if args.use_imagenet:
            model = ResNetOrig(resnet=args.resnet)
        else:
            model.load_state_dict(
                torch.load(args.model_path, map_location=device))
    else:
        model, final_model, results = train(
            args, model, criterion, optimizer, lr_scheduler,
            dataloaders, device)
        test_results = test(args, model, criterion, test_loader, device)
        results.update(test_results)
        results_df = pd.DataFrame(columns=list(results.keys()))
        results_df = results_df.append(results, ignore_index=True)
        results_df.to_parquet(os.path.join(
            exp_dir, f'{args.dataset}_results_{args.iter_idx}.parquet'),
            index=False)
        torch.save(model.state_dict(),
                   os.path.join(
                       exp_dir,
                       f'{args.dataset}_checkpoint_{args.iter_idx}.pt'))
        torch.save(final_model.state_dict(),
                   os.path.join(
                       exp_dir,
                       f'{args.dataset}_checkpoint_final_{args.iter_idx}.pt'))

    # Embed dataset
    if args.save_embeddings:

        os.makedirs(
            os.path.join(constants.DEMO_DATA_DIR, 'train', 'embeddings'),
            exist_ok=True)
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5),
                                 inplace=True),
        ])

        dataset = CXRImageFolder(
            os.path.join(constants.DEMO_DATA_DIR, 'train', args.embed_dataset),
            transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=args.batch_size, num_workers=1, shuffle=False)

        model.fc = torch.nn.Identity()
        model = model.to(device)
        model.eval()

        embeddings_df = pd.DataFrame(
            columns=['Patient ID'] +
                    [f'resnet_dim_{i}' for i in np.arange(2048)])
        pbar = tqdm(dataloader, desc='Inference')
        with torch.no_grad():
            for step, batch in enumerate(pbar):
                batch_results = pd.DataFrame()
                im, label, patient_id = batch
                im = im.to(device)
                result = model(im)
                batch_results['Patient ID'] = np.array(list(patient_id))
                batch_results[[f'resnet_dim_{i}' for i in np.arange(2048)]] = \
                    result.detach().cpu().numpy()
                embeddings_df = embeddings_df.append(batch_results,
                                                     ignore_index=True)
        embeddings_df.to_parquet(os.path.join(
            constants.DEMO_DATA_DIR, 'train', 'embeddings',
            f'embedding_{args.embed_dataset}.parquet'))
