# algorithms/stalesgd.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Tuple
import time
import copy
import heapq # For event queue simulation

# Assuming BaseServer is in server.base_server and BaseClient in clients.base_client
from server.base_server import BaseServer
from utils.simulation import simulate_delay # Using the delay simulation utility
# from clients.base_client import BaseClient # If needed for type hints

class StaleSGDServer(BaseServer):
    """
    Implements the StaleSGD algorithm, specifically the Bounded-Delay Aggregation 
    variant (StaleSGD-BDA).
    
    ASSUMES: Client's train/update method returns the computed GRADIENT, not weights/delta.
    """
    def __init__(self,
                 model: nn.Module,
                 clients: List[Any], # List[BaseClient]
                 test_loader: DataLoader,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initializes the StaleSGDServer (BDA variant).

        Args:
            model (nn.Module): The global model structure.
            clients (List[BaseClient]): A list of client objects.
            test_loader (DataLoader): DataLoader for the global test set.
            config (Dict[str, Any]): Configuration dictionary. Expected keys include:
                                     'max_server_iterations' or 'max_wall_time', 
                                     'eval_interval', 'num_clients',
                                     'server_lr' (eta_g), 'delay_config',
                                     'tau_algo' (Max allowed delay for BDA).
            device (Optional[torch.device]): Device to run the server model on.
        """
        super().__init__(model, clients, test_loader, config, device)
        
        self.num_total_clients = len(clients)
        self.server_lr = config.get('server_lr', 1.0) # eta_g: Server learning rate
        self.delay_config = config.get('delay_config', {'distribution_type': 'exponential', 'params': {'beta': 1.0}})
        self.tau_algo = config.get('tau_algo', self.num_total_clients) # Max delay threshold
        self.max_server_iterations = config.get('max_server_iterations', 100) 
        self.eval_interval = config.get('eval_interval', 10) 

        # --- StaleSGD Specific State ---
        # Server cache (U_i^cache): Stores the latest computed *gradient* from each client
        self.client_gradient_caches: Dict[int, Optional[Dict[str, torch.Tensor]]] = {i: None for i in range(self.num_total_clients)}
        # Stores the server iteration 't' when the model was sent for the cached gradient U_i
        self.client_start_iterations: Dict[int, int] = {i: 0 for i in range(self.num_total_clients)}
        # --- End StaleSGD Specific State ---


        # Simulation state 
        self.client_completion_events = [] # Min-heap: (completion_time, client_id, model_version_sent)
        self.current_wall_time = 0.0
        self.current_server_iteration = 0 # t in the paper's pseudocode

        self.results: Dict[str, List] = { # Standard logging by server iteration
            'server_iteration': [],
            'wall_clock_time': [],
            'test_loss': [],
            'test_accuracy': [],
            'train_loss': [] # Avg loss of gradients used in update t
        }

    def run(self):
        """
        Runs the StaleSGD-BDA simulation loop using an event queue.
        """
        print(f"Starting StaleSGD-BDA simulation...")
        print(f"Total Clients (n): {self.num_total_clients}")
        print(f"Max Delay Threshold (tau_algo): {self.tau_algo}")
        print(f"Server LR: {self.server_lr}")
        print(f"Max server iterations: {self.max_server_iterations}")
        print(f"Delay config: {self.delay_config}")
        print(f"Device: {self.device}")

        self.start_time = time.time()

        # Initialize: All clients compute gradient for w^0
        initial_weights = self.get_model_weights()
        print("Initializing: Sending model w^0 to all clients...")
        for client_id in range(self.num_total_clients):
            self.clients[client_id].set_model_weights(copy.deepcopy(initial_weights))
            self.client_start_iterations[client_id] = 0 # Mark they start with version 0
            
            # Get initial gradient and cache it (simulates initial computation)
            # Assume client returns: gradient, loss, samples
            gradient, _, _ = self.clients[client_id].train() # Mock: Call train, assume it returns gradient

            if self.client_gradient_caches[client_id] is None:
                 # Initialize cache structure if first time
                 self.client_gradient_caches[client_id] = {k: torch.zeros_like(v) for k, v in gradient.items()}
            for name in gradient:
                 self.client_gradient_caches[client_id][name] = gradient[name].cpu().clone()

            # Schedule next completion
            delay = simulate_delay(client_id, **self.delay_config)
            completion_time = self.current_wall_time + delay
            heapq.heappush(self.client_completion_events, (completion_time, client_id, 0)) 
            
        # Initial evaluation (after initial gradients are conceptually computed/cached)
        if self.eval_interval > 0:
             self.log_results(avg_train_loss=None) # Log state at t=0

        # Main simulation loop (driven by events, but updates happen at server steps t)
        # We need a way to trigger server steps. Let's trigger based on time advancing 
        # significantly, or after a certain number of client updates have arrived.
        # A simpler simulation approach: perform server update after every ~N client completions.
        # Let's try triggering server update after N completions for simplicity.
        server_update_trigger_count = self.num_total_clients # Update server after N client responses

        processed_updates_since_last_server_step = 0
        
        while self.current_server_iteration < self.max_server_iterations:
            if not self.client_completion_events:
                print("Warning: No more client events. Stopping simulation.")
                break

            # Process next client completion
            completion_time, client_id, model_version_trained = heapq.heappop(self.client_completion_events)
            self.current_wall_time = completion_time

            # Client completed training - ASSUME it returns gradient
            # Mock structure: gradient, loss, samples = client.get_gradient_update() 
            gradient, loss, samples = self.clients[client_id].train() # Mock: Assume returns gradient

            print(f"Time: {self.current_wall_time:.2f}s | Client {client_id} completed (Trained Ver: {model_version_trained}). Cached gradient updated.")
            
            # Update the cache U_i^cache <- u_i^new (Alg 2, line 5)
            for name in gradient:
                 self.client_gradient_caches[client_id][name] = gradient[name].cpu().clone()
            
            # Immediately relaunch client with current model w^t
            # (Alg 2, line 9 implicitly happens after server step, but client needs work)
            current_weights = self.get_model_weights() # Weights of w^t
            current_iter = self.current_server_iteration
            self.clients[client_id].set_model_weights(copy.deepcopy(current_weights))
            self.client_start_iterations[client_id] = current_iter + 1 # Mark client starts based on w^{t+1} (after potential update)
            
            delay = simulate_delay(client_id, **self.delay_config)
            next_completion_time = self.current_wall_time + delay
            heapq.heappush(self.client_completion_events, (next_completion_time, client_id, current_iter + 1))

            processed_updates_since_last_server_step += 1

            # Check if time to perform a server update step
            # Trigger based on processing roughly N updates, or elapsed time, etc.
            # Let's trigger every N processed updates for this simulation.
            if processed_updates_since_last_server_step >= server_update_trigger_count:
                processed_updates_since_last_server_step = 0 # Reset counter
                t = self.current_server_iteration # Current server step index

                print(f"\n--- Server Update Step {t+1}/{self.max_server_iterations} ---")
                
                # Identify active set A(t) based on staleness <= tau_algo (Alg 2, line 6)
                active_set_indices = [i for i in range(self.num_total_clients) 
                                     if (t + 1) - self.client_start_iterations[i] <= self.tau_algo]
                n_t = len(active_set_indices)
                print(f"  Active set size n_t = {n_t} (Indices: {active_set_indices})")


                if n_t > 0:
                    # Aggregate cached gradients U_i^cache for i in A(t) (Alg 2, line 7)
                    aggregated_gradient_bda = {name: torch.zeros_like(param) 
                                              for name, param in self.client_gradient_caches[active_set_indices[0]].items()}
                    
                    for client_idx in active_set_indices:
                         cached_grad = self.client_gradient_caches[client_idx]
                         for name in aggregated_gradient_bda:
                             aggregated_gradient_bda[name] += cached_grad[name].to(self.device)

                    # Average the gradient over the active set
                    for name in aggregated_gradient_bda:
                        aggregated_gradient_bda[name] = aggregated_gradient_bda[name].float()/float(n_t)

                    # Update Global Model: w^{t+1} = w^t - eta_g * u_BDA^t (Alg 2, line 7)
                    current_weights = self.get_model_weights()
                    with torch.no_grad():
                         new_weights = {name: current_weights[name] - self.server_lr * aggregated_gradient_bda[name] 
                                       for name in current_weights}
                    self.set_model_weights(new_weights)
                    print(f"  Server model updated using {n_t} clients.")
                    
                else:
                    # Skip update if no clients meet staleness criteria (Alg 2, line 8)
                    print(f"  Skipping server update (n_t = 0).")
                    # Model w^{t+1} remains w^t


                self.current_server_iteration += 1 # Increment server iteration

                # Evaluate and Log periodically based on server iterations
                if self.current_server_iteration % self.eval_interval == 0 or self.current_server_iteration == self.max_server_iterations:
                     # Need a way to track representative training loss if desired
                     self.log_results(avg_train_loss=loss) # Using last client's loss as proxy for now
                else:
                     current_time = time.time() - self.start_time
                     print(f"Iter: {self.current_server_iteration:<4} | Time: {current_time:7.2f}s | Last Train Loss: {loss:7.4f}")


            # Optional: Check for wall clock time limit
            max_time = self.config.get('max_wall_time', float('inf'))
            if self.current_wall_time > max_time:
                 print(f"Max wall clock time ({max_time}s) reached. Stopping.")
                 break
                 
        print("\nStaleSGD-BDA simulation finished.")
        if self.current_server_iteration < self.max_server_iterations:
             print(f"Stopped early at server iteration {self.current_server_iteration}")
        if not self.results['server_iteration'] or self.results['server_iteration'][-1] != self.current_server_iteration:
             self.log_results(avg_train_loss=None) # Final evaluation
             
        return self.results