# algorithms/vanilla_asgd.py

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

class VanillaASGDServer(BaseServer):
    """
    Implements the Vanilla Asynchronous Stochastic Gradient Descent (ASGD) algorithm.
    Updates are applied immediately upon arrival from clients.
    """
    def __init__(self,
                 model: nn.Module,
                 clients: List[Any], # List[BaseClient]
                 test_loader: DataLoader,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initializes the VanillaASGDServer.

        Args:
            model (nn.Module): The global model structure.
            clients (List[BaseClient]): A list of client objects.
            test_loader (DataLoader): DataLoader for the global test set.
            config (Dict[str, Any]): Configuration dictionary. Expected keys include:
                                     'max_applied_updates' or 'max_wall_time', 
                                     'eval_interval' (based on applied updates), 
                                     'num_clients', 'concurrency' (Mc),
                                     'server_lr' (eta_g), 'delay_config'.
            device (Optional[torch.device]): Device to run the server model on.
        """
        super().__init__(model, clients, test_loader, config, device)

        self.concurrency = config.get('concurrency', 10) # Mc: Max active clients
        self.server_lr = config.get('server_lr', 1.0) # eta_g: Server learning rate for applying update
        self.delay_config = config.get('delay_config', {'distribution_type': 'exponential', 'params': {'beta': 1.0}})
        self.max_applied_updates = config.get('max_applied_updates', 1000) # Stop after this many updates applied
        self.eval_interval = config.get('eval_interval', 50) # Evaluate every X applied updates

        # Internal state
        self.client_completion_events = [] # Min-heap: (completion_time, client_id, model_version_sent)
        self.active_clients = set() # IDs of clients currently training
        self.idle_clients = list(range(len(clients))) # IDs of clients available
        random.shuffle(self.idle_clients)
        self.current_wall_time = 0.0
        self.applied_update_counter = 0 # Counter for applied client updates
        
        # Track model version conceptually - increments with each applied update
        self.current_model_version = 0 
        self.client_model_versions: Dict[int, int] = {} # Track version sent to client

        # Redefine results structure slightly for clarity
        self.results: Dict[str, List] = {
            'applied_updates': [],
            'wall_clock_time': [],
            'test_loss': [],
            'test_accuracy': [],
            'train_loss': [] 
        }


    def run(self):
        """
        Runs the Vanilla ASGD simulation loop using an event queue.
        """
        print(f"Starting Vanilla ASGD simulation...")
        print(f"Concurrency (Mc): {self.concurrency}")
        print(f"Server LR: {self.server_lr}")
        print(f"Max applied updates: {self.max_applied_updates}")
        print(f"Delay config: {self.delay_config}")
        print(f"Device: {self.device}")

        self.start_time = time.time()

        # Initial evaluation (before any updates)
        if self.eval_interval > 0:
            self.log_asgd_results(avg_train_loss=None) # Log initial state

        # Start initial Mc clients
        initial_weights = self.get_model_weights()
        for _ in range(min(self.concurrency, len(self.idle_clients))):
            client_id = self.idle_clients.pop(0)
            self.active_clients.add(client_id)
            self.clients[client_id].set_model_weights(copy.deepcopy(initial_weights))
            
            # Simulate delay
            delay = simulate_delay(client_id, **self.delay_config)
            completion_time = self.current_wall_time + delay
            heapq.heappush(self.client_completion_events, (completion_time, client_id, 0)) # (time, client_id, model_version)
            self.client_model_versions[client_id] = 0

        # Main simulation loop (event-driven)
        while self.applied_update_counter < self.max_applied_updates:
            if not self.client_completion_events:
                print("Warning: No more client events. Stopping simulation.")
                break

            # Get the next client completion event
            completion_time, client_id, model_version_trained = heapq.heappop(self.client_completion_events)
            
            # Advance simulation time
            self.current_wall_time = completion_time
            
            # Client completed training
            self.active_clients.remove(client_id)
            
            # Get update (assuming client returns delta: w_new - w_old)
            client_model_trained, loss, samples = self.clients[client_id].train() # Assume train returns new weights
            # Get the weights the client started with (approximate for simulation)
            initial_weights_client = self.clients[client_id].model.state_dict() 
            update_delta = {name: client_model_trained[name].cpu() - initial_weights_client[name].cpu() 
                            for name in client_model_trained} # Delta_t^i


            print(f"Time: {self.current_wall_time:.2f}s | Client {client_id} completed (model ver: {model_version_trained}). Applying update {self.applied_update_counter + 1}.")

            # --- Vanilla ASGD Update ---
            # Apply the update immediately: w = w + eta_g * delta_i
            current_weights = self.get_model_weights()
            with torch.no_grad():
                 new_weights = {name: current_weights[name] + self.server_lr * update_delta[name].to(self.device) 
                               for name in current_weights}
            self.set_model_weights(new_weights)
            self.applied_update_counter += 1
            self.current_model_version += 1 # Increment conceptual model version
            # --- End Vanilla ASGD Update ---

            # Evaluate and Log periodically based on applied updates
            if self.applied_update_counter % self.eval_interval == 0 or self.applied_update_counter == self.max_applied_updates:
                 self.log_asgd_results(avg_train_loss=loss) # Log with last client's loss


            # Assign new work to an idle client if available
            new_task_assigned = False
            if self.idle_clients:
                new_client_id = self.idle_clients.pop(0)
                new_task_assigned = True
            else:
                 # If no truly idle clients, reassign the completed client
                 new_client_id = client_id
                 new_task_assigned = True # Reassigning counts as assigning
                 
            if new_task_assigned:
                 self.active_clients.add(new_client_id)
                 current_server_weights = self.get_model_weights() # Send the *very latest* model
                 self.clients[new_client_id].set_model_weights(copy.deepcopy(current_server_weights))
                 
                 delay = simulate_delay(new_client_id, **self.delay_config)
                 next_completion_time = self.current_wall_time + delay
                 
                 next_model_ver = self.current_model_version # Model version the client starts with
                 heapq.heappush(self.client_completion_events, (next_completion_time, new_client_id, next_model_ver))
                 self.client_model_versions[new_client_id] = next_model_ver
                 # print(f"  Assigned Client {new_client_id} with model ver {next_model_ver}. Est completion: {next_completion_time:.2f}s")

            # If the completing client wasn't reassigned (because idle queue had others), add it back
            if not new_task_assigned or new_client_id != client_id:
                 self.idle_clients.append(client_id)


            # Optional: Check for wall clock time limit
            max_time = self.config.get('max_wall_time', float('inf'))
            if self.current_wall_time > max_time:
                 print(f"Max wall clock time ({max_time}s) reached. Stopping.")
                 break
                 
        print("\nVanilla ASGD simulation finished.")
        if self.applied_update_counter < self.max_applied_updates:
             print(f"Stopped early after {self.applied_update_counter} applied updates.")
             if not self.results['applied_updates'] or self.results['applied_updates'][-1] != self.applied_update_counter:
                 self.log_asgd_results(avg_train_loss=None)

        return self.results

    def log_asgd_results(self, avg_train_loss: Optional[float] = None):
         """ Logs evaluation results and timing for ASGD (uses applied_update_counter). """
         test_loss, test_acc = self.evaluate()
         current_time = time.time() - self.start_time
         
         self.results['applied_updates'].append(self.applied_update_counter)
         self.results['wall_clock_time'].append(current_time)
         self.results['test_loss'].append(test_loss)
         self.results['test_accuracy'].append(test_acc)
         self.results['train_loss'].append(avg_train_loss if avg_train_loss is not None else \
                                           (self.results['train_loss'][-1] if self.results['train_loss'] else None))

         print(f"Update: {self.applied_update_counter:<5} | Time: {current_time:7.2f}s | Test Loss: {test_loss:7.4f} | Test Acc: {test_acc:6.2f}% | Last Train Loss: {avg_train_loss if avg_train_loss is not None else 'N/A':7.4f}")