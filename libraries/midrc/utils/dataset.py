"""PyTorch-loadable dataset with MIDRC CXR data."""
import os

import sklearn.model_selection
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class MIDRCDataset(Dataset):
    """MIDRC dataset."""
    def __init__(self, metadata_path, n_samples=None, transform=None, seed=0):
        self.transform = transform

        df = pd.read_csv(metadata_path)
        df.loc[:, 'covid19_positive'] = (
            df['covid19_positive'] == 'Yes').astype(int)

        self.image_paths = df.file_path.values
        self.labels = df.covid19_positive.values

        if n_samples is not None:
            # Randomly select subset of samples, stratified by label
            n_remove = int(len(self.labels) - n_samples)
            idx, _ = sklearn.model_selection.train_test_split(
                list(range(len(self.labels))),
                test_size=n_remove,
                stratify=self.labels,
                random_state=seed,
                shuffle=True,
            )
            self.image_paths = self.image_paths[idx]
            self.labels = self.labels[idx]

        self.image_paths = self.image_paths.tolist()
        self.labels = self.labels.tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_paths[idx])
        image = Image.open(image_path)
        image = image.convert('RGB')
        label = torch.tensor(self.labels[idx]).unsqueeze(-1)
        if self.transform:
            image = self.transform(image)
        return image, label
