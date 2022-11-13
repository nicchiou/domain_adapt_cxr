"""PyTorch-loadable dataset with MIDRC CXR data."""
from sklearn.model_selection import train_test_split
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

class MIDRCDataset(Dataset):
    """MIDRC dataset."""
    def __init__(self, metadata_path, n_samples=None, transform=None, seed=0):
        self.n_samples = n_samples
        self.transform = transform
        self.seed = seed

        self.data = pd.read_csv(metadata_path)
        self.data.loc[:, 'covid19_positive'] = (
            self.data['covid19_positive'] == 'Yes').astype(int)

        images, labels = [], []
        for i in range(len(self.data)):
            image, label = self.__loaditem__(i)
            images.append(image)
            labels.append(label)

        self.images = torch.stack(images, 0)
        self.labels = torch.tensor(labels)

        if n_samples is not None:
            # Randomly select subset of samples, stratified by label
            n_remove = int(len(self.labels) - n_samples)
            idx, _ = train_test_split(
                list(range(len(self.labels))),
                test_size=n_remove,
                stratify=self.labels,
                random_state=seed,
                shuffle=True,
            )
            self.images = self.images[idx]
            self.labels = self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx].unsqueeze(-1)

    def __loaditem__(self, idx):
        image_path = self.data.loc[idx, 'file_path']
        image = Image.open(image_path)
        image = image.convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.data.loc[idx, 'covid19_positive']
        return image, label
