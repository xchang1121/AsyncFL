# clients/base_client.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional, List, Literal, Union
import copy

class BaseClient:
    """
    Base class for a federated learning client.
    Manages local data, model, training process, and communication simulation.
    Modified to support returning weights, deltas, or accumulated gradients.
    """
    def __init__(self,
                 client_id: int,
                 model: nn.Module,
                 dataloader: DataLoader,
                 optimizer_name: str = 'sgd',
                 optimizer_params: Dict[str, Any] = None,
                 loss_fn: nn.Module = None,
                 local_epochs: int = 1,
                 device: Optional[torch.device] = None,
                 # New parameter to control return type
                 return_type: Literal['weights', 'delta', 'gradient'] = 'weights'): 
        """
        Initializes the client.

        Args:
            client_id (int): Unique ID for the client.
            model (nn.Module): The neural network model structure.
            dataloader (DataLoader): DataLoader for the client's local training data.
            optimizer_name (str): Name of the optimizer (e.g., 'sgd', 'adam', 'adamw').
            optimizer_params (Dict[str, Any]): Parameters for the optimizer.
            loss_fn (nn.Module): The loss function.
            local_epochs (int): Number of local epochs to train.
            device (torch.device): The device (CPU or GPU) to train on.
            return_type (str): Specifies what to return after training:
                                'weights': The full state_dict of the trained model.
                                'delta': The difference between trained and initial weights.
                                'gradient': The accumulated gradients during training.
        """
        self.client_id = client_id
        # Store the initial model structure; weights will be set by the server
        self._model_structure = copy.deepcopy(model) 
        self.model = copy.deepcopy(model) # Actual model used for training
        self.dataloader = dataloader
        self.optimizer_name = optimizer_name.lower()
        self.optimizer_params = optimizer_params if optimizer_params is not None else {'lr': 0.01}
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.local_epochs = local_epochs
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.return_type = return_type

        self.model.to(self.device)

    def set_model_weights(self, server_weights: Dict[str, torch.Tensor]):
        """
        Updates the client's local model weights with weights from the server.

        Args:
            server_weights: A state dictionary containing the server model's weights.
        """
        # Ensure the model structure is correct before loading weights
        # This handles cases where the model might have been altered internally
        if not hasattr(self, 'model') or self.model is None:
             self.model = copy.deepcopy(self._model_structure).to(self.device)
             
        self.model.load_state_dict(copy.deepcopy(server_weights))

    def train(self) -> Tuple[Dict[str, torch.Tensor], float, int]:
        """
        Performs local training and returns the specified update type.

        Returns:
            Tuple containing:
            - update (Dict[str, torch.Tensor]): The computed update based on `self.return_type`.
            - average_loss (float): The average training loss over the local epochs.
            - num_samples (int): The number of data samples used for training.
        """
        self.model.train()
        
        # Store initial weights for delta calculation if needed
        initial_weights = None
        if self.return_type == 'delta':
            initial_weights = {name: param.clone().detach() for name, param in self.model.state_dict().items()}
            
        # Initialize structure for accumulating gradients if needed
        accumulated_gradients = None
        if self.return_type == 'gradient':
            accumulated_gradients = {name: torch.zeros_like(param) for name, param in self.model.named_parameters() if param.requires_grad}
            
        # Initialize optimizer
        if self.optimizer_name == 'sgd':
            optimizer = optim.SGD(self.model.parameters(), **self.optimizer_params)
        elif self.optimizer_name == 'adam':
             optimizer = optim.Adam(self.model.parameters(), **self.optimizer_params)
        elif self.optimizer_name == 'adamw':
             optimizer = optim.AdamW(self.model.parameters(), **self.optimizer_params)
        else:
            raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")

        total_loss = 0.0
        total_samples = 0
        total_steps = 0

        for epoch in range(self.local_epochs):
            epoch_samples = 0
            for data, target in self.dataloader:
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = self.model(data)
                loss = self.loss_fn(output, target)
                loss.backward()

                # Accumulate gradients *before* optimizer step if needed
                if self.return_type == 'gradient':
                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            if param.grad is not None:
                                accumulated_gradients[name] += param.grad.clone() # * data.size(0) ? Depends if server expects sum or avg grad

                optimizer.step()
                
                total_loss += loss.item() * data.size(0)
                epoch_samples += data.size(0)
                total_steps += 1 # Count optimizer steps
                
            total_samples = epoch_samples # Use samples from last epoch

        average_loss = total_loss / (self.local_epochs * total_samples) if total_samples > 0 and self.local_epochs > 0 else 0.0

        # Prepare the update based on return_type
        update: Dict[str, torch.Tensor] = {}
        if self.return_type == 'weights':
            update = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
        elif self.return_type == 'delta':
            if initial_weights is None: raise RuntimeError("Initial weights not stored for delta calculation.")
            current_weights = self.model.state_dict()
            update = {name: current_weights[name].cpu().clone() - initial_weights[name].cpu().clone() 
                      for name in initial_weights}
        elif self.return_type == 'gradient':
            if accumulated_gradients is None: raise RuntimeError("Gradients not accumulated.")
            # Decide whether to return sum or average gradient
            # Average gradient seems more standard for aggregation
            update = {name: grad.cpu().clone() / total_steps 
                      for name, grad in accumulated_gradients.items()} if total_steps > 0 else accumulated_gradients
                      
        return update, average_loss, total_samples

    def get_update(self) -> Tuple[Dict[str, torch.Tensor], float, int]:
        """
        Wrapper function to trigger local training and return the results.
        """
        # comp_time = self.simulate_computation_time() # Placeholder
        return self.train()
        
# Example Usage
if __name__ == '__main__':
    # Dummy Model and DataLoader
    dummy_model = nn.Sequential(nn.Linear(10, 5), nn.ReLU(), nn.Linear(5, 2))
    class DummyDataset(torch.utils.data.Dataset):
        def __init__(self, length=100): self.length = length
        def __len__(self): return self.length
        def __getitem__(self, idx): return torch.randn(10), torch.randint(0, 2, (1,)).squeeze()
    dummy_loader = DataLoader(DummyDataset(), batch_size=10)
    
    print("--- Testing 'weights' return type ---")
    client_w = BaseClient(0, dummy_model, dummy_loader, optimizer_params={'lr': 0.1}, local_epochs=1, return_type='weights', device=torch.device('cpu'))
    client_w.set_model_weights(dummy_model.state_dict()) # Set initial weights
    update_w, loss_w, samples_w = client_w.get_update()
    print(f"Update type: {client_w.return_type}, Loss: {loss_w:.4f}, Samples: {samples_w}")
    print(f"Update keys: {list(update_w.keys())}, Example value type: {type(update_w[list(update_w.keys())[0]])}")
    # Check if it matches state_dict keys
    assert all(k in dummy_model.state_dict() for k in update_w.keys())

    print("\n--- Testing 'delta' return type ---")
    client_d = BaseClient(1, dummy_model, dummy_loader, optimizer_params={'lr': 0.1}, local_epochs=1, return_type='delta', device=torch.device('cpu'))
    client_d.set_model_weights(dummy_model.state_dict()) # Set initial weights
    update_d, loss_d, samples_d = client_d.get_update()
    print(f"Update type: {client_d.return_type}, Loss: {loss_d:.4f}, Samples: {samples_d}")
    print(f"Update keys: {list(update_d.keys())}, Example value type: {type(update_d[list(update_d.keys())[0]])}")
    assert all(k in dummy_model.state_dict() for k in update_d.keys())


    print("\n--- Testing 'gradient' return type ---")
    client_g = BaseClient(2, dummy_model, dummy_loader, optimizer_params={'lr': 0.1}, local_epochs=1, return_type='gradient', device=torch.device('cpu'))
    client_g.set_model_weights(dummy_model.state_dict()) # Set initial weights
    update_g, loss_g, samples_g = client_g.get_update()
    print(f"Update type: {client_g.return_type}, Loss: {loss_g:.4f}, Samples: {samples_g}")
    print(f"Update keys: {list(update_g.keys())}, Example value type: {type(update_g[list(update_g.keys())[0]])}")
     # Check if it matches parameter names
    param_names = {name for name, param in dummy_model.named_parameters() if param.requires_grad}
    assert all(k in param_names for k in update_g.keys())