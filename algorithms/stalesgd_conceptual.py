# algorithms/stalesgd_conceptual.py

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

class StaleSGDConceptualServer(BaseServer):
    """
    Implements the Conceptual StaleSGD algorithm (Algorithm 1 from the paper).
    Uses Option A: Direct aggregation of the latest cached gradient from ALL clients.

    ASSUMES: Client's train/update method returns the computed GRADIENT
    """
    def __init__(self,
                 model: nn.Module,
                 clients: List[Any], # List[BaseClient]
                 test_loader: DataLoader,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initializes the StaleSGDConceptualServer (Option A).

        Args:
            model (nn.Module): The global model structure.
            clients (List[BaseClient]): A list of client objects.
            test_loader (DataLoader): DataLoader for the global test set.
            config (Dict[str, Any]): Configuration dictionary. Expected keys include:
                                     'max_server_iterations' or 'max_wall_time',
                                     'eval_interval', 'num_clients',
                                     'server_lr' (eta_g), 'delay_config'.
            device (Optional[torch.device]): Device to run the server model on.
        """
        super().__init__(model, clients, test_loader, config, device)

        self.num_total_clients = len(clients)
        if self.num_total_clients == 0:
            raise ValueError("StaleSGDConceptualServer requires at least one client.")

        self.server_lr = config.get('server_lr', 1.0) # eta_g: Server learning rate
        self.delay_config = config.get('delay_config', {'distribution_type': 'exponential', 'params': {'beta': 1.0}})
        # Use max_server_iterations as primary stopping criterion, matching paper's t
        self.max_server_iterations = config.get('max_server_iterations', 100)
        self.eval_interval = config.get('eval_interval', 10)

        # --- StaleSGD Conceptual (Option A) State ---
        # Server cache (U_i^cache): Stores the latest computed *gradient* from each client
        self.client_gradient_caches: Dict[int, Optional[Dict[str, torch.Tensor]]] = {i: None for i in range(self.num_total_clients)}
        # --- End StaleSGD Conceptual State ---


        # Simulation state
        self.client_completion_events = [] # Min-heap: (completion_time, client_id, model_version_sent_conceptually)
        self.current_wall_time = 0.0
        self.current_server_iteration = 0 # t in the paper's pseudocode

        # Ensure clients are configured to return gradients
        for client in self.clients:
             required_return_type = 'gradient'
             if hasattr(client, 'return_type'):
                  if client.return_type != required_return_type:
                       print(f"Warning: Client {client.client_id} return_type is '{client.return_type}', but StaleSGD Conceptual requires '{required_return_type}'. Attempting to set it.")
                       client.return_type = required_return_type
             else:
                  print(f"Warning: Client {client.client_id} does not have 'return_type' attribute. Assuming it returns gradients.")
                  # If BaseClient structure guarantees gradient return without the flag, this might be okay.


        self.results: Dict[str, List] = { # Standard logging by server iteration
            'server_iteration': [],
            'wall_clock_time': [],
            'test_loss': [],
            'test_accuracy': [],
            'train_loss': [] # Avg loss of the client whose gradient arrived at step t
        }


    def _aggregate_cached_gradients(self) -> Dict[str, torch.Tensor]:
        """
        Helper to compute u^t = (1/n) * sum(U_i^cache)
        Handles potential None values during initialization.
        """
        # Check if all caches are populated. If not (e.g., during init before first round), return zeros.
        if None in self.client_gradient_caches.values():
             print("Warning: Not all client gradient caches are populated. Returning zero gradient.")
             template_model = self.model.state_dict() # Use current model structure
             return {name: torch.zeros_like(param).to(self.device) for name, param in template_model.items()}

        # Get structure from the first valid cache (all should be same structure)
        first_cache = next(iter(self.client_gradient_caches.values()))
        aggregated_gradient = {name: torch.zeros_like(param).to(self.device) for name, param in first_cache.items()}

        for i in range(self.num_total_clients):
            cached_grad = self.client_gradient_caches[i]
            # No need to check for None here due to the check above
            for name in aggregated_gradient:
                # Cache stores CPU tensors, move to server device for aggregation
                aggregated_gradient[name] += cached_grad[name].to(self.device)

        # Average over n (total clients) 
        if self.num_total_clients > 0:
            for name in aggregated_gradient:
                aggregated_gradient[name] /= self.num_total_clients

        return aggregated_gradient


    def run(self):
        """
        Runs the StaleSGD Conceptual simulation loop using an event queue.
        """
        print(f"Starting StaleSGD Conceptual (Option A: Direct Aggregation) simulation...")
        print(f"Total Clients (n): {self.num_total_clients}")
        print(f"Server LR: {self.server_lr}")
        print(f"Max server iterations: {self.max_server_iterations}")
        print(f"Delay config: {self.delay_config}")
        print(f"Device: {self.device}")

        self.start_time = time.time()

        # --- Initialization Step (Lines 1-3 of Alg 1) ---
        initial_weights = self.get_model_weights()
        initial_gradients = {} # Store initial gradients temporarily
        print("Initializing: Sending model w^0 to all clients for initial gradient computation...")

        # 1. Get initial gradient U_i^0 from all clients
        # Simulate this synchronously for simplicity, or use another event loop phase
        initial_agg_grad = None # To store structure
        for client_id in range(self.num_total_clients):
            self.clients[client_id].set_model_weights(copy.deepcopy(initial_weights))
            # Client computes gradient u_i^0 = grad f_i(w^0; xi_i) 
            # Assume train() returns gradient, loss, samples
            gradient_i0, initial_loss, _ = self.clients[client_id].train()
            initial_gradients[client_id] = {k: v.cpu().clone() for k, v in gradient_i0.items()} # Store on CPU
            if initial_agg_grad is None: # Get structure
                 initial_agg_grad = {name: torch.zeros_like(param).to(self.device) for name, param in gradient_i0.items()}


        # 2. Populate cache and compute initial global update u^0
        if initial_agg_grad is None:
             raise RuntimeError("Failed to get gradient structure during initialization.")
             
        for client_id, grad_i0 in initial_gradients.items():
            self.client_gradient_caches[client_id] = grad_i0
            for name in initial_agg_grad:
                 initial_agg_grad[name] += grad_i0[name].to(self.device)

        if self.num_total_clients > 0:
             for name in initial_agg_grad:
                 initial_agg_grad[name] /= self.num_total_clients
        u0 = initial_agg_grad # u^0 computed 

        # 3. Compute w^1 and schedule first completions
        with torch.no_grad():
            weights_w1 = {name: initial_weights[name] - self.server_lr * u0[name]
                          for name in initial_weights}
        self.set_model_weights(weights_w1) # w^1 computed 
        self.current_server_iteration = 0 # Start iteration count after w1 is computed

        # Initial evaluation after w1 is computed
        if self.eval_interval > 0:
            self.log_results(avg_train_loss=None) # Log state at t=0 (represents w1)

        print("Initialization complete. Broadcasting w^1 and starting main loop...")
        # Schedule completion events for all clients based on receiving w^1
        for client_id in range(self.num_total_clients):
             # Send w^1 to client i (already done conceptually by loop completion)
             self.clients[client_id].set_model_weights(copy.deepcopy(weights_w1))
             delay = simulate_delay(client_id, **self.delay_config)
             completion_time = self.current_wall_time + delay # Assuming current time is 0 after init
             heapq.heappush(self.client_completion_events, (completion_time, client_id, 0)) # Client computes based on model iter 0 (w^1)

        # --- Main Simulation Loop (Starts from t=1 in Alg 1, but our counter is 0-based) ---
        # Our loop runs until max_server_iterations updates have been applied
        applied_server_updates = 0
        while applied_server_updates < self.max_server_iterations:
            if not self.client_completion_events:
                print("Warning: No more client events. Stopping simulation.")
                break

            # Get next client completion event (client j_t in Alg 1 notation)  Line 5
            completion_time, client_id, model_version_trained = heapq.heappop(self.client_completion_events)
            self.current_wall_time = completion_time

            # Client j_t completed training based on some older model w_received
            # Client computes u_jt_new = grad f_jt(w_received; xi_new) 
            gradient_jt, loss, samples = self.clients[client_id].train() # Assume returns gradient

            print(f"Time: {self.current_wall_time:.2f}s | Client {client_id} completed. Loss: {loss:.4f}. Applying server update {applied_server_updates + 1}.")

            # --- StaleSGD Conceptual (Option A) Logic Lines 7, 8 ---
            # Update cache: U_jt_cache <- u_jt_new
            # Use deepcopy for safety if gradient tensors might be reused/modified
            for name in gradient_jt:
                 # Ensure cache has the structure initialized
                 if self.client_gradient_caches[client_id] is None:
                     self.client_gradient_caches[client_id] = {k: torch.zeros_like(v) for k, v in gradient_jt.items()}
                 self.client_gradient_caches[client_id][name] = gradient_jt[name].cpu().clone() # Store on CPU

            # Compute u^t = (1/n) * sum(U_i^cache)
            ut = self._aggregate_cached_gradients()
            # --- End Logic ---

            # Update Global Model: w^{t+1} = w^t - eta * u^t  Line 11
            current_weights_t = self.get_model_weights() # This is w^t
            with torch.no_grad():
                 new_weights_tplus1 = {name: current_weights_t[name] - self.server_lr * ut[name].to(self.device)
                                       for name in current_weights_t}
            self.set_model_weights(new_weights_tplus1) # This is now w^{t+1}

            applied_server_updates += 1 # Increment server iteration counter
            self.current_server_iteration = applied_server_updates # Align internal counter

            # Send w^{t+1} back to the client j_t that just finished  Line 11
            self.clients[client_id].set_model_weights(copy.deepcopy(new_weights_tplus1))

            # Schedule next completion for client j_t
            delay = simulate_delay(client_id, **self.delay_config)
            next_completion_time = self.current_wall_time + delay
            heapq.heappush(self.client_completion_events, (next_completion_time, client_id, self.current_server_iteration))

            # Evaluate and Log periodically based on applied server updates
            if applied_server_updates % self.eval_interval == 0 or applied_server_updates == self.max_server_iterations:
                 self.log_results(avg_train_loss=loss) # Use last client's loss as proxy
            else:
                 # Optional: Print intermediate loss
                 current_sim_time = time.time() - self.start_time
                 print(f"  Iter: {applied_server_updates:<4} | Sim Time: {current_sim_time:7.2f}s | Last Train Loss: {loss:7.4f}")


            # Optional: Check for wall clock time limit (using real time)
            max_real_time = self.config.get('max_wall_time_seconds', float('inf'))
            if time.time() - self.start_time > max_real_time:
                 print(f"Max wall clock time ({max_real_time}s) reached. Stopping.")
                 break

        print("\nStaleSGD Conceptual (Option A) simulation finished.")
        if applied_server_updates < self.max_server_iterations:
             print(f"Stopped early after {applied_server_updates} server iterations.")
        # Perform final evaluation if needed and not done
        if not self.results['server_iteration'] or self.results['server_iteration'][-1] != applied_server_updates:
             self.log_results(avg_train_loss=None)

        return self.results