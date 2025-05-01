# clients/base_client.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Any, Tuple, Optional, List, Literal, Union
import copy
import os # Added for cpu_count fallback
import numpy as np # For nan loss handling

class BaseClient:
    """
    Base class for a federated learning client.
    Manages local data, model, training process, and communication simulation.
    Modified to support returning weights, deltas, gradients, or gradient deltas.
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
                 # Extended return type
                 return_type: Literal['weights', 'delta', 'gradient', 'gradient_delta'] = 'weights'):
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
                                'gradient': The accumulated gradients during training (averaged over steps).
                                'gradient_delta': The difference between the new gradient and the previously stored one.
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
        # Initial state for return_type - will be set correctly later
        self._return_type = return_type
        self.previous_gradient: Optional[Dict[str, torch.Tensor]] = None # State for gradient_delta

        self.model.to(self.device)
        # Set initial type using the setter
        self.set_return_type(return_type)


    def set_model_weights(self, server_weights: Dict[str, torch.Tensor]):
        """
        Updates the client's local model weights with weights from the server.

        Args:
            server_weights: A state dictionary containing the server model's weights.
        """
        # Ensure the model structure is correct before loading weights
        if not hasattr(self, 'model') or self.model is None:
             self.model = copy.deepcopy(self._model_structure).to(self.device)

        # Load weights, ensuring they are on the correct device
        try:
            weights_to_load = {k: v.to(self.device) for k, v in server_weights.items()}
            self.model.load_state_dict(weights_to_load)
        except RuntimeError as e:
             print(f"Error loading state_dict for client {self.client_id} (likely key mismatch): {e}")
             print(f"Model keys: {list(self.model.state_dict().keys())}")
             print(f"Received keys: {list(server_weights.keys())}")
             # Decide how to handle: raise error, skip update, etc.
             # For now, we might proceed with the old weights, but this indicates an issue.
             print("Warning: Client model weights not updated due to error.")

    @property
    def return_type(self):
        """Getter for return_type."""
        return self._return_type

    def set_return_type(self, return_type: Literal['weights', 'delta', 'gradient', 'gradient_delta']):
        """Sets the return type and resets related state if necessary."""
        allowed_types = ['weights', 'delta', 'gradient', 'gradient_delta']
        if return_type not in allowed_types:
             raise ValueError(f"Invalid return_type '{return_type}'. Must be one of {allowed_types}")
        self._return_type = return_type
        # Reset previous gradient if the new type is not 'gradient_delta'
        if self._return_type != 'gradient_delta':
            self.previous_gradient = None
        # print(f"Client {self.client_id} return_type set to '{self._return_type}'") # Optional debug print

    def set_previous_gradient(self, gradient: Dict[str, torch.Tensor]):
        """Allows the server to set the initial previous_gradient (needed for Option B). Stores on CPU."""
        # No warning here, assume server knows what it's doing (e.g. during init for Option B)
        # if self.return_type != 'gradient_delta':
        #     print(f"Debug: Setting previous_gradient for client {self.client_id} but return_type is {self.return_type}")
        # Store on CPU
        self.previous_gradient = {k: v.cpu().clone() for k, v in gradient.items()}


    def train(self) -> Tuple[Dict[str, torch.Tensor], float, int]:
        """
        Performs local training and returns the specified update type.
        Handles 'gradient_delta' return type.

        Returns:
            Tuple containing:
            - update (Dict[str, torch.Tensor]): The computed update (CPU tensor). Can be empty dict if training failed/skipped.
            - average_loss (float): The average training loss over the local epochs (NaN if failed/skipped).
            - num_samples (int): The number of data samples in the client's dataset (0 if empty/failed).
        """
        self.model.train()
        # Use internal property _return_type
        current_return_type = self._return_type

        # --- Pre-computation Checks ---
        # Determine dataset size and handle empty dataset case
        dataset_size = 0
        try:
            if hasattr(self.dataloader, 'dataset'):
                 dataset_size = len(self.dataloader.dataset)
            elif hasattr(self.dataloader, '__len__'): # Maybe it's a basic loader
                 dataset_size = len(self.dataloader) * (self.dataloader.batch_size or 1) # Approximation

            if dataset_size == 0:
                print(f"Warning: Client {self.client_id} has an empty dataset or dataloader. Skipping training.")
                # Return empty/zero update
                return {}, float('nan'), 0
        except TypeError:
             print(f"Warning: Could not determine dataset size for client {self.client_id}. Assuming non-empty and proceeding.")
             dataset_size = -1 # Indicate unknown size

        # --- Initialization for Training Step ---
        initial_weights_cpu = None
        if current_return_type == 'delta':
            initial_weights_cpu = {name: param.cpu().clone().detach() for name, param in self.model.state_dict().items()}

        accumulated_gradients_device = None
        model_params = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
        if current_return_type in ['gradient', 'gradient_delta']:
            # Initialize on the client's compute device
            accumulated_gradients_device = {name: torch.zeros_like(param, device=self.device) for name, param in model_params.items()}

        # Initialize optimizer
        try:
            if self.optimizer_name == 'sgd':
                optimizer = optim.SGD(self.model.parameters(), **self.optimizer_params)
            elif self.optimizer_name == 'adam':
                 optimizer = optim.Adam(self.model.parameters(), **self.optimizer_params)
            elif self.optimizer_name == 'adamw':
                 optimizer = optim.AdamW(self.model.parameters(), **self.optimizer_params)
            else:
                raise ValueError(f"Unsupported optimizer: {self.optimizer_name}")
        except Exception as e:
             print(f"Error initializing optimizer for client {self.client_id}: {e}")
             return {}, float('nan'), dataset_size if dataset_size != -1 else 0


        # --- Training Loop ---
        total_loss = 0.0
        total_samples_processed_epoch = 0
        total_steps = 0

        try:
            for epoch in range(self.local_epochs):
                epoch_samples = 0
                for batch_idx, batch in enumerate(self.dataloader):
                    # Basic check for batch content
                    if not isinstance(batch, (list, tuple)) or len(batch) < 2:
                         print(f"Warning: Client {self.client_id} received unexpected batch format: type={type(batch)}, len={len(batch) if isinstance(batch, (list,tuple)) else 'N/A'}. Skipping batch.")
                         continue
                    data, target = batch[0], batch[1]

                    if data.size(0) == 0: continue # Skip empty batches

                    data, target = data.to(self.device), target.to(self.device)

                    optimizer.zero_grad()
                    output = self.model(data)
                    loss = self.loss_fn(output, target)

                    if torch.isnan(loss) or torch.isinf(loss):
                         print(f"Warning: Client {self.client_id} encountered NaN/Inf loss at epoch {epoch+1}, batch {batch_idx}. Skipping batch.")
                         optimizer.zero_grad() # Clear potentially bad gradients
                         continue

                    loss.backward()

                    # Accumulate gradients if needed *before* optimizer step
                    if current_return_type in ['gradient', 'gradient_delta']:
                        with torch.no_grad():
                            for name, param in model_params.items():
                                if param.grad is not None:
                                    if name in accumulated_gradients_device:
                                         accumulated_gradients_device[name] += param.grad # Accumulate on device
                                    else:
                                         print(f"Warning: Gradient accumulated for param '{name}' which was not expected.")

                    optimizer.step()

                    total_loss += loss.item() * data.size(0)
                    epoch_samples += data.size(0)
                    total_steps += 1

                total_samples_processed_epoch = epoch_samples # Track samples from the last epoch run

            # --- Post-Training Processing ---
            if total_steps == 0:
                print(f"Warning: Client {self.client_id} completed training with 0 steps (Dataset size: {dataset_size}).")
                return {}, float('nan'), dataset_size if dataset_size != -1 else 0

            # Calculate average loss
            total_samples_for_loss = total_steps * (self.dataloader.batch_size or 1) # Approx samples processed
            average_loss = total_loss / total_samples_for_loss if total_samples_for_loss > 0 else float('nan')

            # --- Prepare the update (always return CPU tensors) ---
            update: Dict[str, torch.Tensor] = {}
            if current_return_type == 'weights':
                update = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            elif current_return_type == 'delta':
                if initial_weights_cpu is None: raise RuntimeError("Initial weights not stored for delta calculation.")
                current_weights_cpu = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                update = {name: current_weights_cpu[name] - initial_weights_cpu[name]
                          for name in initial_weights_cpu}
            elif current_return_type in ['gradient', 'gradient_delta']:
                if accumulated_gradients_device is None: raise RuntimeError("Gradients not accumulated.")
                # Calculate current average gradient (on CPU)
                current_gradient_cpu = {name: grad.cpu().clone() / total_steps
                                    for name, grad in accumulated_gradients_device.items()}

                if current_return_type == 'gradient':
                    update = current_gradient_cpu
                else: # 'gradient_delta'
                    if self.previous_gradient is None:
                        print(f"Client {self.client_id}: previous_gradient is None. Returning current gradient instead of delta.")
                        update = current_gradient_cpu
                        # Store current gradient as previous for next time (on CPU)
                        self.previous_gradient = current_gradient_cpu
                    else:
                        # Calculate delta (ensure previous is also CPU)
                        update = {name: current_gradient_cpu[name] - self.previous_gradient.get(name, torch.zeros_like(current_gradient_cpu[name]))
                                  for name in current_gradient_cpu}
                        # Crucial: Update previous_gradient for the *next* call (store the new gradient on CPU)
                        self.previous_gradient = current_gradient_cpu
            else:
                 # Should not happen due to initial check in set_return_type
                 raise RuntimeError(f"Unexpected return_type encountered: {current_return_type}")

            # Use actual dataset size if known, otherwise 0
            final_dataset_size = dataset_size if dataset_size != -1 else 0
            return update, average_loss, final_dataset_size

        except Exception as e:
             print(f"!!!!!!!! Error during training for client {self.client_id} !!!!!!!!")
             traceback.print_exc()
             print(f"Error details: {e}")
             print(f"!!!!!!!! Returning empty update for client {self.client_id} !!!!!!!!")
             # Use actual dataset size if known, otherwise 0
             final_dataset_size = dataset_size if dataset_size != -1 else 0
             return {}, float('nan'), final_dataset_size


    def get_update(self) -> Tuple[Dict[str, torch.Tensor], float, int]:
        """
        Wrapper function to trigger local training and return the results.
        """
        return self.train()
