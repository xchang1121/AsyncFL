# algorithms/malenia_sgd.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Tuple
import time
import copy
import heapq # For event queue simulation
import random

# Assuming BaseServer is in server.base_server and BaseClient in clients.base_client
from server.base_server import BaseServer
from utils.simulation import simulate_delay # Using the delay simulation utility
# from clients.base_client import BaseClient # If needed for type hints

class MaleniaSGDServer(BaseServer):
    """
    Implements the Malenia SGD algorithm.
    Ref: Method 6 in https://arxiv.org/pdf/2305.12387.pdf (Malenia SGD Paper)
    
    Note: This implementation uses a simplified stopping condition for the inner loop,
    collecting 'malenia_S' valid gradients per server iteration 'k', similar to
    Rennala SGD (Method 4 in the paper), due to ambiguity in the harmonic mean 
    condition when gradient counts B_i can be zero. 
    
    ASSUMES: Client's train/update method returns the computed GRADIENT, not weights/delta.
    """
    def __init__(self,
                 model: nn.Module,
                 clients: List[Any], # List[BaseClient]
                 test_loader: DataLoader,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initializes the MaleniaSGDServer.

        Args:
            model (nn.Module): The global model structure.
            clients (List[BaseClient]): A list of client objects.
            test_loader (DataLoader): DataLoader for the global test set.
            config (Dict[str, Any]): Configuration dictionary. Expected keys include:
                                     'max_server_iterations' or 'max_wall_time', 
                                     'eval_interval', 'num_clients',
                                     'server_lr' (eta_g), 'delay_config',
                                     'malenia_S' (Batch size S for aggregation).
            device (Optional[torch.device]): Device to run the server model on.
        """
        # Malenia doesn't have explicit concurrency Mc, it uses all n clients
        # But we need a way to manage active computations in simulation.
        # Let's assume all clients are potentially active.
        super().__init__(model, clients, test_loader, config, device)
        
        self.num_total_clients = len(clients)
        self.server_lr = config.get('server_lr', 1.0) # eta_g: Server learning rate
        self.delay_config = config.get('delay_config', {'distribution_type': 'exponential', 'params': {'beta': 1.0}})
        self.malenia_S = config.get('malenia_S', self.num_total_clients) # Default to num_clients if not set
        self.max_server_iterations = config.get('max_server_iterations', 100) 
        self.eval_interval = config.get('eval_interval', 10) 

        # Internal state
        self.client_completion_events = [] # Min-heap: (completion_time, client_id, model_version_sent)
        # Track which model version each client is working on
        self.client_tasks: Dict[int, int] = {} # client_id -> model_version_k they are computing gradient for

        self.current_wall_time = 0.0
        self.current_server_iteration = 0 # k in the paper's pseudocode

        self.results: Dict[str, List] = { # Use standard server iteration logging
            'server_iteration': [],
            'wall_clock_time': [],
            'test_loss': [],
            'test_accuracy': [],
            'train_loss': [] # Avg loss of gradients used in update k
        }


    def run(self):
        """
        Runs the Malenia SGD simulation loop using an event queue.
        """
        print(f"Starting Malenia SGD simulation...")
        print(f"Total Clients (n): {self.num_total_clients}")
        print(f"Gradients per update (S): {self.malenia_S}")
        print(f"Server LR: {self.server_lr}")
        print(f"Max server iterations: {self.max_server_iterations}")
        print(f"Delay config: {self.delay_config}")
        print(f"Device: {self.device}")

        self.start_time = time.time()

        # Initial evaluation
        if self.eval_interval > 0:
            self.log_results(avg_train_loss=None)

        # Start all n clients
        initial_weights = self.get_model_weights()
        for client_id in range(self.num_total_clients):
            self.clients[client_id].set_model_weights(copy.deepcopy(initial_weights))
            self.client_tasks[client_id] = 0 # Assigned to compute gradient for model w^0
            
            delay = simulate_delay(client_id, **self.delay_config)
            completion_time = self.current_wall_time + delay
            heapq.heappush(self.client_completion_events, (completion_time, client_id, 0)) # (time, client_id, model_version_sent)

        # Main simulation loop (per server iteration k)
        for k in range(self.max_server_iterations):
            self.current_server_iteration = k
            print(f"\n--- Server Iteration {k+1}/{self.max_server_iterations} ---")
            
            valid_gradients_for_k = [] # Store (gradient, loss, samples)
            
            # Inner loop: Collect S valid gradients for iteration k
            while len(valid_gradients_for_k) < self.malenia_S:
                if not self.client_completion_events:
                    print("Warning: No more client events but required gradients not met. Stopping.")
                    # Need to handle this case - maybe break outer loop or proceed with fewer gradients?
                    # For now, break outer loop.
                    self.current_server_iteration = self.max_server_iterations 
                    break 

                # Get next completion event
                completion_time, client_id, model_version_trained = heapq.heappop(self.client_completion_events)
                self.current_wall_time = completion_time

                # Client completed training - ASSUME it returns gradient
                # IMPORTANT: Modify client `train` or `get_update` to return gradient
                # Mock structure: gradient, loss, samples = client.get_gradient_update() 
                
                # Mocking client return: gradient, loss, samples
                # In reality, need client to calculate and return gradient
                client_model_trained, loss, samples = self.clients[client_id].train() # Simulate training
                # Calculate gradient (e.g., average gradient if multiple steps) - Mock: use delta as proxy gradient
                initial_weights_client = self.clients[client_id].model.state_dict() # Approx. starting weights
                gradient = {name: (initial_weights_client[name].cpu() - client_model_trained[name].cpu()) / self.clients[client_id].optimizer_params.get('lr', 0.1) 
                           for name in client_model_trained} # Mock: delta / lr 


                print(f"Time: {self.current_wall_time:.2f}s | Client {client_id} completed (Trained Ver: {model_version_trained}). Current Server Ver: {k}")

                # Check if gradient is valid for current iteration k (Malenia Line 8: k' == k)
                if model_version_trained == k:
                    print(f"  Gradient is valid for iteration {k}. Adding to batch ({len(valid_gradients_for_k)+1}/{self.malenia_S}). Loss: {loss:.4f}")
                    valid_gradients_for_k.append((gradient, loss, samples))
                else:
                    # Gradient is stale w.r.t. current server iter k, ignore it (Malenia Line 8: else clause implied)
                    print(f"  Gradient is stale (for ver {model_version_trained}). Discarding.")

                # Relaunch the client with the *current* model (w^k) for the current iteration k
                # (Malenia Line 11: Send (x^k, k) to the worker)
                current_weights = self.get_model_weights() # Get weights of w^k
                self.clients[client_id].set_model_weights(copy.deepcopy(current_weights))
                self.client_tasks[client_id] = k # Assign to compute gradient for model w^k
                
                delay = simulate_delay(client_id, **self.delay_config)
                next_completion_time = self.current_wall_time + delay
                heapq.heappush(self.client_completion_events, (next_completion_time, client_id, k)) # Client starts working on version k

            # Break outer loop if inner loop broke early
            if self.current_server_iteration == self.max_server_iterations:
                break
                
            # Aggregate the S valid gradients (simple average, Malenia Line 14 uses weighted avg by 1/Bi?)
            # Paper Line 14: g^k = (1/n) * sum( (1/Bi) * sum(gradients_from_i) ). This is complex.
            # Rennala (Method 4) uses g^k = (1/S) * sum(received_valid_gradients). Let's use this simpler average.
            aggregated_gradient = {name: torch.zeros_like(param) 
                                   for name, param in valid_gradients_for_k[0][0].items()}
            total_loss_in_batch = 0.0
            total_samples_in_batch = 0
            
            for grad, loss, samples in valid_gradients_for_k:
                 for name in aggregated_gradient:
                     aggregated_gradient[name] += grad[name].to(self.device)
                 total_loss_in_batch += loss * samples
                 total_samples_in_batch += samples
            
            # Average the gradient
            for name in aggregated_gradient:
                 aggregated_gradient[name] /= self.malenia_S
                 
            avg_train_loss_k = total_loss_in_batch / total_samples_in_batch if total_samples_in_batch > 0 else 0.0

            # Update Global Model: w^{k+1} = w^k - eta_g * g^k (Malenia Line 15)
            current_weights = self.get_model_weights()
            with torch.no_grad():
                 new_weights = {name: current_weights[name] - self.server_lr * aggregated_gradient[name] 
                               for name in current_weights}
            self.set_model_weights(new_weights)
            
            print(f"  Server model updated for iteration {k+1}.")

            # Evaluate and Log periodically
            if (k + 1) % self.eval_interval == 0 or (k + 1) == self.max_server_iterations:
                self.log_results(avg_train_loss=avg_train_loss_k)
            else:
                 current_time = time.time() - self.start_time
                 print(f"Iter: {k+1:<4} | Time: {current_time:7.2f}s | Avg Train Loss (k): {avg_train_loss_k:7.4f}")


            # Optional: Check for wall clock time limit
            max_time = self.config.get('max_wall_time', float('inf'))
            if self.current_wall_time > max_time:
                 print(f"Max wall clock time ({max_time}s) reached. Stopping.")
                 break

        print("\nMalenia SGD simulation finished.")
        # Final evaluation if needed
        if self.current_server_iteration < self.max_server_iterations:
             print(f"Stopped early at server iteration {self.current_server_iteration}")
        if not self.results['server_iteration'] or self.results['server_iteration'][-1] != self.current_server_iteration:
             self.log_results(avg_train_loss=None)
             
        return self.results

    # Use standard log_results based on server iterations
    # log_results = BaseServer.log_results