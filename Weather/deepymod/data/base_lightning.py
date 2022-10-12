""" Contains the base class for the Dataset (1 and 2 dimensional) and a function
     that takes a Pytorch tensor and converts it to a numpy array"""
from typing import Any

import torch
import numpy as np
from numpy import ndarray
from numpy.random import default_rng
from deepymod.data.samples import Subsampler

from abc import ABC, ABCMeta, abstractmethod
import pandas as pd

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):

        self.coords = data
        self.target = labels

    def __getitem__(self, idx: int):
        """Returns coordinate and value. First axis of coordinate should be time."""
        return self.coords[idx], self.target[idx]

    def __len__(self):
        return self.target.shape[0]

