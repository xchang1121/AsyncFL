# datasets/data_utils.py

import random
import numpy as np
from typing import List, Dict, Tuple, Union
# Optional: Import torch if directly handling PyTorch datasets
# from torch.utils.data import Dataset 

def partition_data_dirichlet(
    data_indices: Union[np.ndarray, List[int]], 
    labels: Union[np.ndarray, List[int]], 
    num_clients: int, 
    alpha: float, 
    num_classes: int = None,
    seed: int = 42
) -> Dict[int, List[int]]:
    """
    Partitions data indices among clients based on label distribution using a 
    Dirichlet distribution.

    Args:
        data_indices: A list or numpy array of data indices to partition.
        labels: A list or numpy array of corresponding labels for each data index.
                Should be the same length as data_indices.
        num_clients: The number of clients to partition the data for.
        alpha: The concentration parameter for the Dirichlet distribution. 
               Smaller alpha leads to higher heterogeneity (non-IID).
        num_classes: The total number of unique classes in the dataset. If None, 
                     it's inferred from the labels.
        seed: Random seed for reproducibility.

    Returns:
        A dictionary where keys are client IDs (0 to num_clients-1) and 
        values are lists of data indices assigned to that client.
    """
    np.random.seed(seed)
    
    if len(data_indices) != len(labels):
        raise ValueError("data_indices and labels must have the same length.")

    if num_classes is None:
        num_classes = len(np.unique(labels))
        
    if num_clients <= 0:
        raise ValueError("Number of clients must be positive.")
        
    if alpha <= 0:
         raise ValueError("Alpha must be positive for Dirichlet distribution.")

    # Ensure labels are numpy array for efficient indexing
    labels = np.array(labels)
    data_indices = np.array(data_indices)

    # Get indices for each class
    class_indices = [np.where(labels == i)[0] for i in range(num_classes)]
    
    # Check if any class has no samples - Dirichlet partitioning might fail
    for i, indices in enumerate(class_indices):
        if len(indices) == 0:
            print(f"Warning: Class {i} has no samples in the provided data indices.")
            
    # client_proportions[i, j] = proportion of class j samples assigned to client i
    client_proportions = np.random.dirichlet([alpha] * num_clients, num_classes)

    client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}
    
    for c in range(num_classes):
        class_c_indices = class_indices[c]
        if len(class_c_indices) == 0:
            continue # Skip classes with no data
            
        # Shuffle indices within the class for random assignment within proportions
        np.random.shuffle(class_c_indices)
        
        # Calculate cumulative proportions for assignment
        proportions = client_proportions[c]
        
        # Ensure proportions sum to 1 (handle potential floating point issues)
        proportions = proportions / proportions.sum()
        
        # Calculate number of samples per client for this class
        samples_per_client = (np.cumsum(proportions) * len(class_c_indices)).astype(int)
        
        # Correct potential rounding errors ensuring total samples match
        samples_per_client[-1] = len(class_c_indices) 
        
        current_idx = 0
        for client_id in range(num_clients):
            num_samples = samples_per_client[client_id]
            # Handle potential empty slices due to rounding or small class size
            if num_samples > current_idx : 
                 assigned_indices = class_c_indices[current_idx:num_samples]
                 # Use original data_indices corresponding to the selected class indices
                 client_indices[client_id].extend(data_indices[assigned_indices].tolist())
                 current_idx = num_samples
            # If num_samples <= current_idx, it means this client gets 0 samples of this class
            # which can happen with small alpha and few samples per class.
            # Ensure current_idx is updated even if no samples assigned in this round
            elif client_id > 0 :
                 samples_per_client[client_id] = samples_per_client[client_id-1]


    # Shuffle indices within each client's list
    for client_id in client_indices:
        random.shuffle(client_indices[client_id])

    return client_indices

# Example Usage (can be removed or placed in a test file)
if __name__ == '__main__':
    # Simulate data: 1000 data points, 10 classes
    N_SAMPLES = 1000
    N_CLASSES = 10
    indices = np.arange(N_SAMPLES)
    # Create a somewhat balanced label distribution for demonstration
    sim_labels = np.array([i % N_CLASSES for i in range(N_SAMPLES)]) 
    
    N_CLIENTS = 10
    
    # --- Highly Non-IID Example ---
    alpha_non_iid = 0.1
    print(f"\n--- Partitioning with alpha = {alpha_non_iid} (Highly Non-IID) ---")
    client_data_non_iid = partition_data_dirichlet(indices, sim_labels, N_CLIENTS, alpha_non_iid, N_CLASSES)
    
    print(f"Number of clients: {len(client_data_non_iid)}")
    total_samples_assigned = 0
    for client_id, client_idxs in client_data_non_iid.items():
        print(f"Client {client_id}: {len(client_idxs)} samples")
        total_samples_assigned += len(client_idxs)
        # Check label distribution for a few clients
        if client_id < 3: 
             client_labels = sim_labels[client_idxs]
             label_counts = np.unique(client_labels, return_counts=True)
             print(f"  Label distribution: {dict(zip(label_counts[0], label_counts[1]))}")
    print(f"Total samples assigned: {total_samples_assigned} (Expected: {N_SAMPLES})")

    # --- Closer to IID Example ---
    alpha_iid = 100
    print(f"\n--- Partitioning with alpha = {alpha_iid} (Closer to IID) ---")
    client_data_iid = partition_data_dirichlet(indices, sim_labels, N_CLIENTS, alpha_iid, N_CLASSES)

    print(f"Number of clients: {len(client_data_iid)}")
    total_samples_assigned_iid = 0
    for client_id, client_idxs in client_data_iid.items():
        print(f"Client {client_id}: {len(client_idxs)} samples")
        total_samples_assigned_iid += len(client_idxs)
        # Check label distribution for a few clients
        if client_id < 3:
             client_labels = sim_labels[client_idxs]
             label_counts = np.unique(client_labels, return_counts=True)
             print(f"  Label distribution: {dict(zip(label_counts[0], label_counts[1]))}")
    print(f"Total samples assigned: {total_samples_assigned_iid} (Expected: {N_SAMPLES})")