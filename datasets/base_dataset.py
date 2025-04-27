# datasets/base_dataset.py

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple
from torch.utils.data import DataLoader, Dataset

class BaseDataset(ABC):
    """
    Abstract base class for datasets used in the federated learning simulation.
    Provides a common interface for loading, partitioning, and accessing data.
    """
    def __init__(self, data_root: str = './data'):
        """
        Initializes the BaseDataset.

        Args:
            data_root (str): The root directory where data is stored or will be downloaded.
        """
        self.data_root = data_root
        self.train_dataset: Any = None
        self.test_dataset: Any = None
        self.client_indices: Dict[int, List[int]] = {}

    @abstractmethod
    def load_data(self) -> None:
        """
        Loads the training and testing datasets into memory or prepares them
        for access. Must populate self.train_dataset and self.test_dataset.
        """
        pass

    @abstractmethod
    def partition_data(self, num_clients: int, **kwargs) -> None:
        """
        Partitions the training data among the specified number of clients.
        Must populate self.client_indices.

        Args:
            num_clients (int): The number of clients to partition data for.
            **kwargs: Additional arguments specific to the partitioning strategy 
                      (e.g., alpha for Dirichlet).
        """
        pass
        
    @abstractmethod
    def get_train_dataloader(self, client_id: int, batch_size: int, **kwargs) -> DataLoader:
        """
        Returns a DataLoader for the training data assigned to a specific client.

        Args:
            client_id (int): The ID of the client.
            batch_size (int): The batch size for the DataLoader.
             **kwargs: Additional arguments for DataLoader (e.g., num_workers).


        Returns:
            DataLoader: A DataLoader for the client's training data.
            
        Raises:
            ValueError: If data has not been partitioned or client_id is invalid.
        """
        pass

    @abstractmethod
    def get_test_dataloader(self, batch_size: int, **kwargs) -> DataLoader:
        """
        Returns a DataLoader for the global test dataset.

        Args:
            batch_size (int): The batch size for the DataLoader.
            **kwargs: Additional arguments for DataLoader (e.g., num_workers).

        Returns:
            DataLoader: A DataLoader for the test data.
            
        Raises:
             ValueError: If test data has not been loaded.
        """
        pass

    @property
    def num_classes(self) -> int:
        """
        Returns the number of classes in the dataset. Needs to be implemented 
        or set by subclasses after loading data.
        
        Returns:
            int: Number of classes.
            
        Raises:
            NotImplementedError: If not implemented by the subclass.
            AttributeError: If the attribute holding the number of classes is not set.
        """
        raise NotImplementedError("Subclasses must implement the num_classes property.")