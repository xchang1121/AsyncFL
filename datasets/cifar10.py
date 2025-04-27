# datasets/cifar10.py

import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from typing import Dict, List, Any, Optional

from .base_dataset import BaseDataset
from .data_utils import partition_data_dirichlet

class CIFAR10Dataset(BaseDataset):
    """
    Handles loading, partitioning, and serving of the CIFAR-10 dataset.
    """
    def __init__(self, data_root: str = './data'):
        super().__init__(data_root)
        self._num_classes = 10
        self.mean = (0.4914, 0.4822, 0.4465) 
        self.std = (0.2023, 0.1994, 0.2010)
        # Alternative normalization: (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    @property
    def num_classes(self) -> int:
        return self._num_classes

    def load_data(self) -> None:
        """
        Downloads and loads the CIFAR-10 dataset using torchvision.
        """
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        self.train_dataset = datasets.CIFAR10(
            self.data_root, train=True, download=True, transform=transform_train
        )
        self.test_dataset = datasets.CIFAR10(
            self.data_root, train=False, download=True, transform=transform_test
        )
        print("CIFAR-10 dataset loaded.")

    def partition_data(self, num_clients: int, alpha: float = 0.5, seed: int = 42) -> None:
        """
        Partitions the CIFAR-10 training data using Dirichlet distribution.

        Args:
            num_clients (int): The number of clients.
            alpha (float): Concentration parameter for Dirichlet distribution.
            seed (int): Random seed.
        """
        if self.train_dataset is None:
            self.load_data()

        if not hasattr(self.train_dataset, 'targets'):
             # Handle datasets that might not have a .targets attribute directly
             # For CIFAR10 from torchvision, this should exist.
             try:
                 # Attempt to get labels for common dataset structures
                 labels = np.array([sample[1] for sample in self.train_dataset])
             except Exception as e:
                  raise AttributeError(f"Could not extract labels from train_dataset. Error: {e}")
        else:
              labels = np.array(self.train_dataset.targets)
              
        if len(labels) == 0:
             raise ValueError("Could not extract labels from the training dataset.")


        indices = np.arange(len(self.train_dataset))
        
        self.client_indices = partition_data_dirichlet(
            data_indices=indices,
            labels=labels,
            num_clients=num_clients,
            alpha=alpha,
            num_classes=self._num_classes,
            seed=seed
        )
        print(f"CIFAR-10 data partitioned for {num_clients} clients with alpha={alpha}.")


    def get_train_dataloader(self, client_id: int, batch_size: int, **kwargs) -> DataLoader:
        """
        Gets the training DataLoader for a specific client.
        """
        if not self.client_indices:
            raise ValueError("Data has not been partitioned. Call partition_data first.")
        if client_id not in self.client_indices:
            raise ValueError(f"Invalid client_id: {client_id}")
        if self.train_dataset is None:
             raise ValueError("Training data not loaded.")

        client_subset_indices = self.client_indices[client_id]
        if not client_subset_indices:
             print(f"Warning: Client {client_id} has no data assigned.")
             # Return an empty DataLoader or handle as appropriate
             return DataLoader(Subset(self.train_dataset, []), batch_size=batch_size, **kwargs)
             
        subset = Subset(self.train_dataset, client_subset_indices)
        
        # Common kwargs: num_workers, pin_memory, shuffle=True
        kwargs.setdefault('shuffle', True) 
        
        dataloader = DataLoader(subset, batch_size=batch_size, **kwargs)
        return dataloader

    def get_test_dataloader(self, batch_size: int, **kwargs) -> DataLoader:
        """
        Gets the global test DataLoader.
        """
        if self.test_dataset is None:
            raise ValueError("Test data not loaded. Call load_data first.")
            
        # Common kwargs: num_workers, pin_memory, shuffle=False
        kwargs.setdefault('shuffle', False)
        
        dataloader = DataLoader(self.test_dataset, batch_size=batch_size, **kwargs)
        return dataloader

# Example Usage
if __name__ == '__main__':
    cifar_data = CIFAR10Dataset(data_root='./data')
    
    # Load data
    cifar_data.load_data()
    print(f"Number of classes: {cifar_data.num_classes}")
    
    # Partition data
    N_CLIENTS = 100
    ALPHA = 0.5
    cifar_data.partition_data(num_clients=N_CLIENTS, alpha=ALPHA)
    
    # Get DataLoader for client 0
    try:
        train_loader_client_0 = cifar_data.get_train_dataloader(client_id=0, batch_size=32)
        print(f"\nSuccessfully created train DataLoader for client 0.")
        # Optionally iterate through a batch
        # for data, target in train_loader_client_0:
        #     print(f"Client 0: Batch data shape: {data.shape}, Batch target shape: {target.shape}")
        #     break 
    except ValueError as e:
        print(f"Error creating DataLoader for client 0: {e}")

    # Get test DataLoader
    test_loader = cifar_data.get_test_dataloader(batch_size=128)
    print(f"Successfully created test DataLoader.")
    # Optionally iterate through a batch
    # for data, target in test_loader:
    #     print(f"Test: Batch data shape: {data.shape}, Batch target shape: {target.shape}")
    #     break