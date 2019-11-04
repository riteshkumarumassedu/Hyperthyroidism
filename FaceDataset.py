import torch
import torch.nn as nn
import os
import torch.optim as optim
from skimage import io, transform
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from torch.utils.data import sampler

import torchvision.datasets as dset
from torchvision import transforms, utils
import torchvision.transforms as T
import torch.nn.functional as F
import pandas as pd
import numpy as np

class FaceDataset(Dataset):
    """Face dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.imgData = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.imgData)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir,
                                self.imgData.iloc[idx, 0] + '.jpg')
        image = io.imread(img_name)
#         image = image.transpose((2, 0, 1))
#         print(image.shape)
#         image.view(3, image.shape[0], image.shape[1])
        label = self.imgData.iloc[idx, 1]
#         landmarks = landmarks.astype('float').reshape(-1, 2)
        sample = {'image': image, 'label': label}

#         sample = {'image': image, 'label': label, 'name':img_name}

        if self.transform:
            sample = self.transform(sample)

        return sample