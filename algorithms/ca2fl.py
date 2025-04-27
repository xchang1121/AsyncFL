# algorithms/ca2fl.py

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

class CA2FLServer(BaseServer):
    """
    Implements the Cache-Aided Asynchronous Federated Learning (CA2FL) algorithm.
    Ref: Algorithm 2 in https://openreview.net/pdf?id=0_csAAaS7L (CA2FL Paper)
    """
    def __init__(self,
                 model: nn.Module,
                 clients: List[Any], # List[BaseClient]
                 test_loader: DataLoader,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initializes the CA2FLServer.

        Args:
            model (nn.Module): The global model structure.
            clients (List[BaseClient]): A list of client objects.
            test_loader (DataLoader): DataLoader for the global test set.
            config (Dict[str, Any]): Configuration dictionary. Expected keys include:
                                     'max_server_iterations' or 'max_wall_time', 
                                     'eval_interval', 'num_clients',
                                     'concurrency' (Mc), 'buffer_size_k' (M in paper),
                                     'server_lr' (eta_g), 'delay_config'.
            device (Optional[torch.device]): Device to run the server model on.
        """
        super().__init__(model, clients, test_loader, config, device)

        self.concurrency = config.get('concurrency', 10) # Mc: Max active clients
        self.buffer_size_k = config.get('buffer_size_k', 10) # M: Updates needed for server step
        self.server_lr = config.get('server_lr', 1.0) # eta_g: Server learning rate
        self.delay_config = config.get('delay_config', {'distribution_type': 'exponential', 'params': {'beta': 1.0}})

        # --- CA2FL Specific State ---
        self.num_total_clients = len(clients)
        # Client cache (h_i): Stores the last received *delta* from each client
        self.client_caches_h_i: Dict[int, Optional[Dict[str, torch.Tensor]]] = {i: None for i in range(self.num_total_clients)}
        # Global cache (h): Average of client caches
        self.global_cache_h: Optional[Dict[str, torch.Tensor]] = None
        # Buffer for *calibrated* updates: (delta_i - h_i_prev, num_samples)
        self.calibrated_update_buffer: List[Tuple[Dict[str, torch.Tensor], int]] = [] 
        # Set of clients contributing to the current buffer
        self.clients_in_buffer: set[int] = set() 
        # Initialize caches with zeros
        with torch.no_grad():
            template_state_dict = self.model.state_dict()
            zero_delta = {name: torch.zeros_like(param) for name, param in template_state_dict.items()}
            for i in range(self.num_total_clients):
                 self.client_caches_h_i[i] = copy.deepcopy(zero_delta)
            self.global_cache_h = copy.deepcopy(zero_delta)
        # --- End CA2FL Specific State ---


        # Simulation state (similar to FedBuff)
        self.client_completion_events = [] # Min-heap: (completion_time, client_id, model_version_sent)
        self.active_clients = set() # IDs of clients currently training
        self.idle_clients = list(range(len(clients))) # IDs of clients available
        random.shuffle(self.idle_clients)
        self.current_wall_time = 0.0
        self.server_update_counter = 0 # Counter for server steps based on buffer flushes
        self.client_model_versions: Dict[int, int] = {}


    def run(self):
        """
        Runs the asynchronous CA2FL simulation loop using an event queue.
        """
        print(f"Starting CA2FL simulation...")
        print(f"Concurrency (Mc): {self.concurrency}")
        print(f"Buffer Size (M): {self.buffer_size_k}")
        print(f"Server LR: {self.server_lr}")
        print(f"Max server iterations (updates): {self.max_server_iterations}")
        print(f"Delay config: {self.delay_config}")
        print(f"Device: {self.device}")

        self.start_time = time.time()

        # Initial evaluation
        if self.eval_interval > 0:
             self.log_results(avg_train_loss=None)

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
        while self.server_update_counter < self.max_server_iterations:
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

            print(f"Time: {self.current_wall_time:.2f}s | Client {client_id} completed (model ver: {model_version_trained}). Loss: {loss:.4f}. Samples: {samples}")

            # --- CA2FL Specific Logic ---
            # Get previous cache for this client
            h_i_prev = self.client_caches_h_i[client_id]
            if h_i_prev is None: # Should not happen after initialization
                 raise RuntimeError(f"Cache for client {client_id} not initialized.")
                 
            # Calculate calibrated update: delta_i - h_i_prev (Algorithm 2, line 4)
            calibrated_delta = {name: update_delta[name] - h_i_prev[name] for name in update_delta}

            # Add calibrated update to buffer
            if samples > 0:
                self.calibrated_update_buffer.append((calibrated_delta, samples))
                self.clients_in_buffer.add(client_id) # Track clients in this buffer batch
                print(f"  Calibrated update added. Buffer size: {len(self.calibrated_update_buffer)}/{self.buffer_size_k}")

            # Update client's cache h_i: h_{t+1}^i = Delta_t^i (Algorithm 2, line 5)
            self.client_caches_h_i[client_id] = copy.deepcopy(update_delta) 
            # --- End CA2FL Specific Logic ---

            # Trigger server update if buffer is full
            if len(self.calibrated_update_buffer) >= self.buffer_size_k:
                print(f"  Buffer full. Performing server update {self.server_update_counter + 1}...")
                
                # --- CA2FL Aggregation (Algorithm 2, line 10, 11) ---
                # Calculate average calibrated delta from buffer
                avg_calibrated_delta = {name: torch.zeros_like(param) 
                                       for name, param in self.calibrated_update_buffer[0][0].items()}
                num_in_buffer = len(self.calibrated_update_buffer)
                for cal_delta, _ in self.calibrated_update_buffer:
                    for name in avg_calibrated_delta:
                        avg_calibrated_delta[name] += cal_delta[name].to(self.device)
                for name in avg_calibrated_delta:
                     avg_calibrated_delta[name] /= num_in_buffer
                     
                # Calculate server update vt = ht + avg_calibrated_delta (implicitly Alg 2 line 10, 11, 12)
                server_update_v = {name: self.global_cache_h[name].to(self.device) + avg_calibrated_delta[name]
                                   for name in self.global_cache_h}
                # --- End CA2FL Aggregation ---

                # Update global model: w_{t+1} = w_t + eta_g * v_t (Algorithm 2, line 12)
                current_weights = self.get_model_weights()
                with torch.no_grad():
                     new_weights = {name: current_weights[name] + self.server_lr * server_update_v[name] 
                                   for name in current_weights}
                self.set_model_weights(new_weights)
                
                # --- Update Global Cache h (Algorithm 2, line 13, 14) ---
                # h_{t+1} = (1/n) * sum(h_{t+1}^i)
                # Note: h_{t+1}^i was already updated for clients in self.clients_in_buffer
                # For others, h_{t+1}^i = h_t^i (which is already stored in self.client_caches_h_i)
                temp_global_h = {name: torch.zeros_like(param) 
                                 for name, param in self.global_cache_h.items()}
                for i in range(self.num_total_clients):
                     for name in temp_global_h:
                         temp_global_h[name] += self.client_caches_h_i[i][name] # Adds h_{t+1}^i or h_t^i
                
                self.global_cache_h = {name: param / self.num_total_clients 
                                        for name, param in temp_global_h.items()}
                # --- End Update Global Cache ---


                self.server_update_counter += 1
                self.calibrated_update_buffer = [] # Clear buffer
                self.clients_in_buffer = set() # Reset clients contributing to buffer


                print(f"  Server model updated. Iteration: {self.server_update_counter}")
                
                # Evaluate and Log periodically based on server updates
                if self.server_update_counter % self.eval_interval == 0 or self.server_update_counter == self.max_server_iterations:
                     self.log_results(avg_train_loss=loss) # Using last client's loss as proxy for now


            # Assign new work to an idle client if available (same as FedBuff)
            if self.idle_clients:
                new_client_id = self.idle_clients.pop(0)
                self.active_clients.add(new_client_id)
                current_server_weights = self.get_model_weights()
                self.clients[new_client_id].set_model_weights(copy.deepcopy(current_server_weights))
                
                delay = simulate_delay(new_client_id, **self.delay_config)
                completion_time = self.current_wall_time + delay
                
                current_model_ver = self.server_update_counter 
                heapq.heappush(self.client_completion_events, (completion_time, new_client_id, current_model_ver))
                self.client_model_versions[new_client_id] = current_model_ver
                print(f"  Assigned Client {new_client_id} with model ver {current_model_ver}. Est completion: {completion_time:.2f}s")

            else:
                self.idle_clients.append(client_id)


            # Optional: Check for wall clock time limit
            max_time = self.config.get('max_wall_time', float('inf'))
            if self.current_wall_time > max_time:
                 print(f"Max wall clock time ({max_time}s) reached. Stopping.")
                 break
                 
        print("\nCA2FL simulation finished.")
        if self.server_update_counter < self.max_server_iterations:
             print(f"Stopped early at server iteration {self.server_update_counter}")
             if not self.results['server_iteration'] or self.results['server_iteration'][-1] != self.server_update_counter:
                  self.log_results(avg_train_loss=None)

        return self.results