# algorithms/delay_adaptive.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Tuple
import time
import copy
import heapq # For event queue simulation
import random
import math

# Assuming BaseServer is in server.base_server and BaseClient in clients.base_client
from server.base_server import BaseServer
from utils.simulation import simulate_delay # Using the delay simulation utility
# from clients.base_client import BaseClient # If needed for type hints

class DelayAdaptiveAFLServer(BaseServer):
    """
    Implements an Asynchronous SGD algorithm with delay-adaptive learning rates.
    The learning rate for an update is scaled down based on its staleness,
    inspired by Koloskova et al. 2022 (https://arxiv.org/abs/2206.08307).
    """
    def __init__(self,
                 model: nn.Module,
                 clients: List[Any], # List[BaseClient]
                 test_loader: DataLoader,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initializes the DelayAdaptiveAFLServer.

        Args:
            model (nn.Module): The global model structure.
            clients (List[BaseClient]): A list of client objects.
            test_loader (DataLoader): DataLoader for the global test set.
            config (Dict[str, Any]): Configuration dictionary. Expected keys include:
                                     'max_applied_updates' or 'max_wall_time', 
                                     'eval_interval' (based on applied updates), 
                                     'num_clients', 'concurrency' (Mc),
                                     'server_lr' (base eta), 'delay_config',
                                     'staleness_threshold_tau_c' (optional, defaults to concurrency).
            device (Optional[torch.device]): Device to run the server model on.
        """
        super().__init__(model, clients, test_loader, config, device)

        self.concurrency = config.get('concurrency', 10) # Mc: Max active clients
        self.base_server_lr = config.get('server_lr', 1.0) # Base eta
        self.delay_config = config.get('delay_config', {'distribution_type': 'exponential', 'params': {'beta': 1.0}})
        self.max_applied_updates = config.get('max_applied_updates', 1000) # Stop after this many updates applied
        self.eval_interval = config.get('eval_interval', 50) # Evaluate every X applied updates
        
        # Threshold for staleness adaptation (tau_C in the paper)
        # Defaults to concurrency Mc if not specified
        self.staleness_threshold_tau_c = float(config.get('staleness_threshold_tau_c', self.concurrency)) 
        if self.staleness_threshold_tau_c <= 0:
             print(f"Warning: staleness_threshold_tau_c should be positive. Setting to concurrency {self.concurrency}.")
             self.staleness_threshold_tau_c = float(self.concurrency)


        # Internal state (similar to VanillaASGD)
        self.client_completion_events = [] # Min-heap: (completion_time, client_id, model_version_sent)
        self.active_clients = set() # IDs of clients currently training
        self.idle_clients = list(range(len(clients))) # IDs of clients available
        random.shuffle(self.idle_clients)
        self.current_wall_time = 0.0
        self.applied_update_counter = 0 # Counter for applied client updates
        self.current_model_version = 0 
        self.client_model_versions: Dict[int, int] = {} # Track version sent to client

        self.results: Dict[str, List] = { # Use same results structure as VanillaASGD
            'applied_updates': [],
            'wall_clock_time': [],
            'test_loss': [],
            'test_accuracy': [],
            'train_loss': [] 
        }


    def _get_adaptive_lr(self, staleness: int) -> float:
        """
        Calculates the adaptive learning rate based on staleness.
        Implements a variation of Eq 11 from Koloskova et al. 2022.
        If staleness > tau_c, scales base_lr by tau_c / staleness.

        Args:
            staleness (int): The delay (in model versions) of the update.

        Returns:
            float: The adaptive learning rate eta_t.
        """
        if staleness <= self.staleness_threshold_tau_c:
            return self.base_server_lr
        else:
            # Scale down LR for higher staleness. Clip at base_lr.
            # Alternative: Could use min(eta, 1/(4*L*staleness)) if L is known/estimated
            # Alternative: Could use 0 (drop update)
            adaptive_lr = self.base_server_lr * (self.staleness_threshold_tau_c / staleness)
            return min(self.base_server_lr, adaptive_lr) # Ensure it doesn't exceed base LR


    def run(self):
        """
        Runs the Delay-Adaptive ASGD simulation loop using an event queue.
        """
        print(f"Starting Delay-Adaptive ASGD simulation...")
        print(f"Concurrency (Mc): {self.concurrency}")
        print(f"Base Server LR (eta): {self.base_server_lr}")
        print(f"Staleness Threshold (tau_C): {self.staleness_threshold_tau_c}")
        print(f"Max applied updates: {self.max_applied_updates}")
        print(f"Delay config: {self.delay_config}")
        print(f"Device: {self.device}")

        self.start_time = time.time()

        # Initial evaluation
        if self.eval_interval > 0:
            self.log_asgd_results(avg_train_loss=None)

        # Start initial Mc clients
        initial_weights = self.get_model_weights()
        for _ in range(min(self.concurrency, len(self.idle_clients))):
            client_id = self.idle_clients.pop(0)
            self.active_clients.add(client_id)
            self.clients[client_id].set_model_weights(copy.deepcopy(initial_weights))
            
            delay = simulate_delay(client_id, **self.delay_config)
            completion_time = self.current_wall_time + delay
            heapq.heappush(self.client_completion_events, (completion_time, client_id, 0)) 
            self.client_model_versions[client_id] = 0

        # Main simulation loop
        while self.applied_update_counter < self.max_applied_updates:
            if not self.client_completion_events:
                print("Warning: No more client events. Stopping simulation.")
                break

            completion_time, client_id, model_version_trained = heapq.heappop(self.client_completion_events)
            self.current_wall_time = completion_time
            self.active_clients.remove(client_id)
            
            client_model_trained, loss, samples = self.clients[client_id].train()
            initial_weights_client = self.clients[client_id].model.state_dict()
            update_delta = {name: client_model_trained[name].cpu() - initial_weights_client[name].cpu() 
                            for name in client_model_trained}

            # --- Delay-Adaptive Specific Logic ---
            staleness = self.current_model_version - model_version_trained
            adaptive_lr = self._get_adaptive_lr(staleness)
            # --- End Delay-Adaptive Specific Logic ---
            
            print(f"Time: {self.current_wall_time:.2f}s | Client {client_id} completed (Stale: {staleness}, Ver: {model_version_trained}). LR: {adaptive_lr:.4f}. Applying update {self.applied_update_counter + 1}.")

            # Apply update with adaptive LR: w = w + eta_t * delta_i
            current_weights = self.get_model_weights()
            with torch.no_grad():
                 new_weights = {name: current_weights[name] + adaptive_lr * update_delta[name].to(self.device) 
                               for name in current_weights}
            self.set_model_weights(new_weights)
            self.applied_update_counter += 1
            self.current_model_version += 1

            # Evaluate and Log periodically
            if self.applied_update_counter % self.eval_interval == 0 or self.applied_update_counter == self.max_applied_updates:
                 self.log_asgd_results(avg_train_loss=loss)

            # Assign new work (same logic as VanillaASGD)
            new_task_assigned = False
            if self.idle_clients:
                new_client_id = self.idle_clients.pop(0)
                new_task_assigned = True
            else:
                 new_client_id = client_id
                 new_task_assigned = True 
                 
            if new_task_assigned:
                 self.active_clients.add(new_client_id)
                 current_server_weights = self.get_model_weights() # Send latest model
                 self.clients[new_client_id].set_model_weights(copy.deepcopy(current_server_weights))
                 
                 delay = simulate_delay(new_client_id, **self.delay_config)
                 next_completion_time = self.current_wall_time + delay
                 
                 next_model_ver = self.current_model_version 
                 heapq.heappush(self.client_completion_events, (next_completion_time, new_client_id, next_model_ver))
                 self.client_model_versions[new_client_id] = next_model_ver
                 # print(f"  Assigned Client {new_client_id} with model ver {next_model_ver}. Est completion: {next_completion_time:.2f}s")

            if not new_task_assigned or new_client_id != client_id:
                 self.idle_clients.append(client_id)

            # Optional: Check for wall clock time limit
            max_time = self.config.get('max_wall_time', float('inf'))
            if self.current_wall_time > max_time:
                 print(f"Max wall clock time ({max_time}s) reached. Stopping.")
                 break
                 
        print("\nDelay-Adaptive ASGD simulation finished.")
        if self.applied_update_counter < self.max_applied_updates:
             print(f"Stopped early after {self.applied_update_counter} applied updates.")
             if not self.results['applied_updates'] or self.results['applied_updates'][-1] != self.applied_update_counter:
                 self.log_asgd_results(avg_train_loss=None)

        return self.results

    # Use the same logging function as VanillaASGD
    log_asgd_results = VanillaASGDServer.log_asgd_results