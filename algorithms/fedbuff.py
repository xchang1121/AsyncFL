# algorithms/fedbuff.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Tuple
import time
import copy
import random
import heapq # For event queue simulation

# Assuming BaseServer is in server.base_server and BaseClient in clients.base_client
from server.base_server import BaseServer
from utils.simulation import simulate_delay # Using the delay simulation utility
# from clients.base_client import BaseClient # If needed for type hints

class FedBuffServer(BaseServer):
    """
    Implements the Federated Learning with Buffered Asynchronous Aggregation (FedBuff) algorithm.
    Ref: Algorithm 1 in https://openreview.net/pdf?id=0_csAAaS7L (CA2FL Paper)
         Section 3 in https://proceedings.mlr.press/v151/nguyen22a/nguyen22a.pdf (FedBuff Paper)
    """
    def __init__(self,
                 model: nn.Module,
                 clients: List[Any], # List[BaseClient]
                 test_loader: DataLoader,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initializes the FedBuffServer.

        Args:
            model (nn.Module): The global model structure.
            clients (List[BaseClient]): A list of client objects.
            test_loader (DataLoader): DataLoader for the global test set.
            config (Dict[str, Any]): Configuration dictionary. Expected keys include:
                                     'max_server_iterations' or 'max_wall_time', 
                                     'eval_interval', 'num_clients',
                                     'concurrency' (Mc), 'buffer_size_k' (K or M in papers),
                                     'server_lr' (eta_g), 'delay_config' (for simulation).
            device (Optional[torch.device]): Device to run the server model on.
        """
        super().__init__(model, clients, test_loader, config, device)

        self.concurrency = config.get('concurrency', 10) # Mc: Max active clients
        self.buffer_size_k = config.get('buffer_size_k', 10) # K (or M): Updates needed for server step
        self.server_lr = config.get('server_lr', 1.0) # eta_g: Server learning rate
        self.delay_config = config.get('delay_config', {'distribution_type': 'exponential', 'params': {'beta': 1.0}})

        # Internal state
        self.update_buffer: List[Tuple[Any, int]] = [] # Stores (update_delta, num_samples)
        self.client_completion_events = [] # Min-heap: (completion_time, client_id, model_version_sent)
        self.active_clients = set() # IDs of clients currently training
        self.idle_clients = list(range(len(clients))) # IDs of clients available
        random.shuffle(self.idle_clients)
        self.current_wall_time = 0.0
        self.current_server_iteration = 0 # Counter for server steps based on buffer flushes

        # Track model version sent to each client
        self.client_model_versions: Dict[int, int] = {}


    def aggregate(self, updates: List[Tuple[Dict[str, torch.Tensor], int]], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Aggregates updates (deltas) in the buffer for FedBuff.
        Overrides BaseServer.aggregate.

        Args:
            updates (List[Tuple[Dict[str, torch.Tensor], int]]): List of updates from buffer. 
                                       Each update is (update_delta, num_samples).
            **kwargs: Can include 'current_weights'.

        Returns:
            Dict[str, torch.Tensor]: The new model weights (state_dict).
        """
        if not updates:
            return self.get_model_weights()

        # FedBuff aggregates the deltas/updates directly (Alg 1, line 10 & 14 [cite: 72])
        # Simple averaging of the deltas in the buffer.
        aggregated_delta = None
        
        # Initialize aggregated_delta with zeros based on the first delta's structure
        first_delta = updates[0][0]
        aggregated_delta = {name: torch.zeros_like(param) for name, param in first_delta.items()}

        num_updates_in_buffer = len(updates)
        
        for delta, _ in updates: # FedBuff paper suggests simple average (1/K)
            for name in aggregated_delta:
                 aggregated_delta[name] += delta[name].to(self.device)
        
        # Average the aggregated delta
        for name in aggregated_delta:
            aggregated_delta[name] = aggregated_delta[name].float() / num_updates_in_buffer


        # Apply the aggregated delta to the current model weights
        current_weights = kwargs.get('current_weights', self.get_model_weights())
        new_weights = copy.deepcopy(current_weights)
        with torch.no_grad():
             for name in new_weights:
                  # Note: FedBuff applies server_lr * aggregated_delta. 
                  # If client update = w_new - w_old, server update is w = w + sg_lr * (1/K * sum(w_new_i - w_old_i))
                  # If client update = gradient, server update is w = w - sg_lr * (1/K * sum(grad_i))
                  # Assuming client returns delta w_new - w_old:
                  # Update rule w_t+1 = w_t + eta_g * avg(delta_i)
                  # Since aggregate() returns NEW weights, we compute w_t + delta here.
                  # Let's redefine aggregate to return the averaged_delta instead.
                  pass # Redefining below

        # --- Redefinition: Aggregate returns the averaged delta ---
        averaged_delta = {name: torch.zeros_like(param) for name, param in updates[0][0].items()}
        for delta, _ in updates:
             for name in averaged_delta:
                 averaged_delta[name] += delta[name].to(self.device)
        
        if num_updates_in_buffer > 0:
            for name in averaged_delta:
                averaged_delta[name] = averaged_delta[name].float() / num_updates_in_buffer
                
        return averaged_delta # Return the averaged delta


    def run(self):
        """
        Runs the asynchronous FedBuff simulation loop using an event queue.
        """
        print(f"Starting FedBuff simulation...")
        print(f"Concurrency (Mc): {self.concurrency}")
        print(f"Buffer Size (K): {self.buffer_size_k}")
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
            
            # Simulate delay for this client's first round
            delay = simulate_delay(client_id, **self.delay_config)
            completion_time = self.current_wall_time + delay
            heapq.heappush(self.client_completion_events, (completion_time, client_id, 0)) # (time, client_id, model_version)
            self.client_model_versions[client_id] = 0

        # Main simulation loop (event-driven)
        while self.current_server_iteration < self.max_server_iterations:
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
            # TODO: Ensure client side returns the delta, version number, loss, samples
            # update_delta, loss, samples = self.clients[client_id].get_update_delta() 
            # For now, let's mock this return value structure
            initial_weights_client = copy.deepcopy(self.clients[client_id].model.state_dict())
            client_model_trained, loss, samples = self.clients[client_id].train() # Assume train returns new weights
            # initial_weights_client = self.clients[client_id].model.state_dict() # Approximation of weights client started with
            update_delta = {name: client_model_trained[name].cpu() - initial_weights_client[name].cpu() 
                            for name in client_model_trained}


            print(f"Time: {self.current_wall_time:.2f}s | Client {client_id} completed (model ver: {model_version_trained}). Loss: {loss:.4f}. Samples: {samples}")

            # Add update to buffer
            if samples > 0:
                self.update_buffer.append((update_delta, samples))
                print(f"  Update buffer size: {len(self.update_buffer)}/{self.buffer_size_k}")


            # Trigger server update if buffer is full
            if len(self.update_buffer) >= self.buffer_size_k:
                print(f"  Buffer full. Performing server update {self.current_server_iteration + 1}...")
                
                averaged_delta = self.aggregate(self.update_buffer) # Gets the averaged delta

                # Update global model: w_new = w_old + eta_g * averaged_delta
                current_weights = self.get_model_weights()
                with torch.no_grad():
                     new_weights = {name: current_weights[name] + self.server_lr * averaged_delta[name].to(self.device) 
                                   for name in current_weights}
                self.set_model_weights(new_weights)

                self.current_server_iteration += 1
                self.update_buffer = [] # Clear buffer

                print(f"  Server model updated. Iteration: {self.current_server_iteration}")
                
                 # Evaluate and Log periodically based on server updates
                if self.current_server_iteration % self.eval_interval == 0 or self.current_server_iteration == self.max_server_iterations:
                     # Need a way to track avg train loss across buffer flushes if desired
                     self.log_results(avg_train_loss=loss) # Using last client's loss as proxy for now


            
            if not self.idle_clients:
                # If no idle clients, the completed client becomes idle immediately
                self.idle_clients.append(client_id)

            # Assign new work to an idle client if available
            new_client_id = self.idle_clients.pop(0)
            self.active_clients.add(new_client_id)
            current_server_weights = self.get_model_weights()
            self.clients[new_client_id].set_model_weights(copy.deepcopy(current_server_weights))
            
            # Simulate delay for the new task
            delay = simulate_delay(new_client_id, **self.delay_config)
            completion_time = self.current_wall_time + delay
            
            current_model_ver = self.current_server_iteration # Model version the client starts with
            heapq.heappush(self.client_completion_events, (completion_time, new_client_id, current_model_ver))
            self.client_model_versions[new_client_id] = current_model_ver
            print(f"  Assigned Client {new_client_id} with model ver {current_model_ver}. Est completion: {completion_time:.2f}s")
                


            # Optional: Check for wall clock time limit
            max_time = self.config.get('max_wall_time', float('inf'))
            if self.current_wall_time > max_time:
                 print(f"Max wall clock time ({max_time}s) reached. Stopping.")
                 break

        print("\nFedBuff simulation finished.")
        if self.current_server_iteration < self.max_server_iterations:
             print(f"Stopped early at server iteration {self.current_server_iteration}")
             # Perform final evaluation if needed and not done
             if self.results['server_iteration'][-1] != self.current_server_iteration:
                  self.log_results(avg_train_loss=None)


        return self.results