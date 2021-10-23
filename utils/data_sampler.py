import numpy as np
import torch
from sklearn.model_selection import train_test_split


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
