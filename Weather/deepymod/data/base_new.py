""" Contains the base class for the Dataset (1 and 2 dimensional) and a function
     that takes a Pytorch tensor and converts it to a numpy array"""
from typing import Any

import torch
import numpy as np
from numpy import ndarray
from numpy.random import default_rng
from deepymod.data.samples import Subsampler

from abc import ABC, ABCMeta, abstractmethod


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data, labels, shuffle=None,
                 subsampler: Subsampler = None,
                 load_kwargs: dict = {},
                 device: str = None,
                 ):
        """A dataset class that loads the data, preprocesses it and lastly applies subsampling to it

        Args:
            load_function (func):Must return torch tensors in the format coordinates, data
            shuffle (bool, optional): Shuffle the data. Defaults to True.
            apply_normalize (func)
            subsampler (Subsampler, optional): Add some subsampling function. Defaults to None.
            load_kwargs (dict, optional): kwargs to pass to the load_function. Defaults to {}.
            preprocess_kwargs (dict, optional): (optional) arguments to pass to the preprocess method. Defaults to {
            "random_state": 42, "noise_level": 0.0, "normalize_coords": False, "normalize_data": False, }.
            subsampler_kwargs (dict, optional): (optional) arguments to pass to the subsampler method. Defaults to {}.
            device (str, optional): which device to send the data to. Defaults to None.
        """
        self.subsampler = subsampler
        self.load_kwargs = load_kwargs
        self.device = device
        self.shuffle = shuffle

        self.coords = data
        self.data = labels
        # Ensure the data that loaded is not 0D/1D
        assert (
                len(self.coords.shape) >= 2
        ), "Please explicitely specify a feature axis for the coordinates"
        assert (
                len(self.data.shape) >= 2
        ), "Please explicitely specify a feature axis for the data"

        # Now we know the data are shape (number_of_samples, number_of_features) we can set the number_of_samples
        self.number_of_samples = self.data.shape[0]

        print("Dataset is using device: ", self.device)
        if self.device:
            self.coords = self.coords.to(self.device)
            self.data = self.data.to(self.device)

    # Pytorch methods
    def __len__(self) -> int:
        """Returns length of dataset. Required by pytorch"""
        return self.data.shape[0]

    def __getitem__(self, idx: int) -> tuple[Any, Any]:
        """Returns coordinate and value. First axis of coordinate should be time."""
        return self.coords[idx], self.data[idx]

    # get methods
    def get_coords(self):
        """Retrieve all the coordinate features"""
        return self.coords

    def get_data(self):
        """Retrieve all the data features"""
        return self.data


def small_pipeline(data, labels,device, loader_params: dict = {}):

    dataset = Dataset(data=data, labels=labels, device=device)
    dataloader = torch.utils.data.DataLoader(dataset, **loader_params)

    return dataloader
