import gzip
import os

import torch
import torch.nn as nn
from torch.nn import Conv2d, LeakyReLU, Dropout, MaxPool2d, Linear
from torch.nn.modules import Flatten
from utils import initialize_weights


class BaseModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name

    def save(self, save_dir_path: str) -> None:
        """
        Save the model's weights to be later restored
        """
        with gzip.open(os.path.join(save_dir_path, self.model_name), "wb") as out_file:
            torch.save(self.state_dict(), out_file)


class MnistClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv1 -> LeakyReLU -> Dropout 0.2
        # Conv2 -> LeakyReLU -> MaxPool -> Dropout 0.2 -> Flatten
        # FC1 -> LeakyReLU -> Dropout 0.2 -> FC2
        self.model = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # (32, 28, 28)
            LeakyReLU(),
            Dropout(0.2),

            Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # (64, 28, 28)
            LeakyReLU(),
            MaxPool2d(kernel_size=2, stride=2),  # (64, 14, 14)
            Dropout(0.2),
            Flatten(),

            Linear(64 * 14 * 14, 128),
            LeakyReLU(),
            Dropout(0.2),

            Linear(128, 10)
        )

        self.apply(initialize_weights)

    def forward(self, x):
        """
        Input:
        - x: (None, 1, 28, 28)

        Output:
        - logits (None, 10) - NO SOFTMAX
        """
        return self.model(x)
