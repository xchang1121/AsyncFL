# algorithms/stalesgd_conceptual.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Tuple, Literal
import time
import copy
import heapq
import random
import numpy as np
import os
import traceback

# Assuming BaseServer is in server.base_server
from server.base_server import BaseServer
from utils.simulation import simulate_delay
# Assumes modified BaseClient is available in clients.base_client
from clients.base_client import BaseClient # Import the specific class for type hinting/checks


class StaleSGDConceptualServer(BaseServer):
    """
    Implements Conceptual StaleSGD (Algorithm 1 from the paper 'Provable Benefit...'),
    supporting both implementation options:
    - Option A: Direct aggregation (server cache U_i^cache).
    - Option B: Incremental update (server stores u^{t-1}, client sends delta).

    Requires modified BaseClient supporting 'gradient' and 'gradient_delta' return types.
    """
    def __init__(self,
                 model: nn.Module,
                 clients: List[BaseClient], # Expect list of BaseClient instances
                 test_loader: DataLoader,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initializes the StaleSGDConceptualServer.

        Args:
            model (nn.Module): The global model structure.
            clients (List[BaseClient]): A list of client objects.
            test_loader (DataLoader): DataLoader for the global test set.
            config (Dict[str, Any]): Configuration dictionary. Expected keys include:
                                     'max_server_iterations', 'eval_interval',
                                     'server_lr', 'delay_config',
                                     'conceptual_stalegsgd_option' ('A' or 'B').
            device (Optional[torch.device]): Device to run the server model on.
        """
        super().__init__(model, clients, test_loader, config, device)

        self.num_total_clients = len(clients)
        if self.num_total_clients == 0:
            raise ValueError("StaleSGDConceptualServer requires at least one client.")

        self.server_lr = config.get('server_lr', 1.0) # eta_g
        self.delay_config = config.get('delay_config', {'distribution_type': 'exponential', 'params': {'beta': 1.0}})
        # Use max_server_iterations as primary stopping criterion, matching paper's t
        self.max_server_iterations = config.get('max_server_iterations', 100)
        self.eval_interval = config.get('eval_interval', 10)

        self.option = config.get('conceptual_stalegsgd_option', 'A').upper()
        if self.option not in ['A', 'B']:
            raise ValueError("conceptual_stalegsgd_option in config must be 'A' or 'B'")

        print(f"Initializing StaleSGD Conceptual with Option {self.option}")

        # --- State Specific to Options ---
        self.client_gradient_caches: Dict[int, Optional[Dict[str, torch.Tensor]]] = {} # Option A state (CPU tensors)
        self.previous_aggregated_update: Optional[Dict[str, torch.Tensor]] = None    # Option B state (CPU tensor)

        # Determine required client return types
        self._client_return_type = 'gradient_delta' if self.option == 'B' else 'gradient'
        self._client_init_return_type = 'gradient' # Always need initial gradient

        # Initialize state variables and check/set client types
        if self.option == 'A':
            self.client_gradient_caches = {i: None for i in range(self.num_total_clients)}

        # Check clients and set initial type
        for client in self.clients:
             if not isinstance(client, BaseClient):
                  raise TypeError(f"Client list contains non-BaseClient object: {client}")
             try:
                # Set type needed for init phase; will be changed later for Option B if necessary
                client.set_return_type(self._client_init_return_type)
             except AttributeError as e:
                 print(f"Error: Client {client.client_id} is likely not the modified BaseClient (missing set_return_type).")
                 raise RuntimeError("Clients must be instances of the modified BaseClient.") from e


        # --- Simulation state (common) ---
        self.client_completion_events = [] # Min-heap: (completion_time, client_id, model_version_sent_conceptually)
        self.current_wall_time = 0.0
        self.current_server_iteration = 0 # t (tracks number of server updates applied)

        # Standard results logging
        self.results: Dict[str, List] = {
            'server_iteration': [],
            'wall_clock_time': [],
            'test_loss': [],
            'test_accuracy': [],
            'train_loss': [], # Avg loss of the client whose update arrived
        }

    # --- Helper methods ---
    def _get_initial_gradients(self, initial_weights) -> Dict[int, Dict[str, torch.Tensor]]:
        """Helper to get initial gradients u_i^0 from all clients. Returns CPU tensors."""
        initial_gradients = {}
        print("Getting initial gradients u_i^0 from all clients...")

        # Ensure clients are set to return 'gradient' for this phase
        # (Already done in __init__)

        for client_id in range(self.num_total_clients):
            client = self.clients[client_id]
            # Make sure return type is gradient for this step
            if client.return_type != self._client_init_return_type:
                 client.set_return_type(self._client_init_return_type)

            client.set_model_weights(copy.deepcopy(initial_weights)) # Send w^0
            try:
                 gradient_i0, _, _ = client.get_update() # Returns CPU tensors
                 if not gradient_i0: # Handle empty dataset case or training failure
                      print(f"Warning: Client {client_id} returned empty initial gradient. Using zeros.")
                      gradient_i0 = {name: torch.zeros_like(param).cpu()
                                     for name, param in self.model.named_parameters() if param.requires_grad}
                 initial_gradients[client_id] = gradient_i0 # Already CPU tensors
            except Exception as e:
                 print(f"Error getting initial gradient from client {client_id}: {e}")
                 traceback.print_exc()
                 gradient_i0 = {name: torch.zeros_like(param).cpu()
                                for name, param in self.model.named_parameters() if param.requires_grad}
                 initial_gradients[client_id] = gradient_i0

        print("Initial gradients u_i^0 collected.")
        return initial_gradients

    def _aggregate_option_a(self) -> Dict[str, torch.Tensor]:
        """Computes u^t = (1/n) * sum(U_i^cache) for Option A. Returns tensor on server device."""
        expected_keys = {p[0] for p in self.model.named_parameters() if p[1].requires_grad}
        if not expected_keys: raise RuntimeError("Model has no trainable parameters.")
        template_param = next(iter(self.model.parameters()))

        aggregated_gradient = {}
        with torch.no_grad(): # Initialize zeros on the correct device
             for name, param in self.model.named_parameters():
                  if name in expected_keys:
                       aggregated_gradient[name] = torch.zeros_like(param).to(self.device)

        valid_caches_count = 0
        for i in range(self.num_total_clients):
            cached_grad = self.client_gradient_caches.get(i)
            if cached_grad is not None and isinstance(cached_grad, dict):
                 valid_caches_count +=1
                 for name in expected_keys:
                     if name in cached_grad:
                        try:
                            aggregated_gradient[name] += cached_grad[name].to(self.device)
                        except Exception as e:
                             print(f"Error adding cached grad from client {i} for key {name}: {e}. Skipping key.")

        if valid_caches_count < self.num_total_clients:
             print(f"Warning: Aggregated gradients from {valid_caches_count}/{self.num_total_clients} clients (some caches might be initializing/missing).")

        if self.num_total_clients > 0:
            for name in aggregated_gradient:
                aggregated_gradient[name] /= self.num_total_clients
        return aggregated_gradient

    def _update_option_b(self, gradient_delta_jt: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Computes u^t = u^{t-1} + delta_jt / n for Option B. Returns tensor on server device."""
        if self.previous_aggregated_update is None:
             raise RuntimeError("Option B Error: previous_aggregated_update is None during update step.")

        new_aggregated_update_device = {}
        for name in self.previous_aggregated_update:
             prev_val_device = self.previous_aggregated_update[name].to(self.device)
             delta_val_device = gradient_delta_jt.get(name, torch.zeros_like(prev_val_device)).to(self.device)

             if self.num_total_clients > 0:
                new_aggregated_update_device[name] = prev_val_device + delta_val_device / self.num_total_clients
             else:
                 new_aggregated_update_device[name] = prev_val_device

        self.previous_aggregated_update = {k: v.cpu().clone() for k, v in new_aggregated_update_device.items()}
        return new_aggregated_update_device

    # --- Main Execution Logic ---
    def run(self):
        """Runs the StaleSGD Conceptual simulation loop using an event queue."""
        print(f"--- Starting StaleSGD Conceptual (Option {self.option}) ---")
        print(f"Num Clients: {self.num_total_clients}, Server LR: {self.server_lr}, Max Iter: {self.max_server_iterations}")
        self.start_time = time.time()

        # --- Initialization (Algorithm 1, Lines 1-3) ---
        initial_weights = self.get_model_weights()
        initial_gradients = self._get_initial_gradients(initial_weights)

        # Calculate u^0
        if not initial_gradients or not any(initial_gradients.values()):
             print("Warning: No valid initial gradients collected. Initializing u^0 to zeros.")
             u0_struct = {name: param for name, param in self.model.named_parameters() if param.requires_grad}
             u0 = {name: torch.zeros_like(param).to(self.device) for name, param in u0_struct.items()}
        else:
             u0_struct = next(g for g in initial_gradients.values() if g)
             u0 = {name: torch.zeros_like(param).to(self.device) for name, param in u0_struct.items()}
             for i in range(self.num_total_clients):
                  grad_i0 = initial_gradients.get(i)
                  if grad_i0:
                       for name in u0:
                           if name in grad_i0: u0[name] += grad_i0[name].to(self.device)
             if self.num_total_clients > 0:
                  for name in u0: u0[name] /= self.num_total_clients

        # Initialize state based on option
        if self.option == 'A':
            self.client_gradient_caches = initial_gradients # Store initial CPU gradients
        else: # Option B
            self.previous_aggregated_update = {k: v.cpu().clone() for k, v in u0.items()} # Store u^0 on CPU
            print("Option B: Sending initial gradients u_i^0 back to clients for u_i^prev storage.")
            for client_id, grad_i0 in initial_gradients.items():
                if grad_i0:
                     try: self.clients[client_id].set_previous_gradient(grad_i0)
                     except AttributeError: raise RuntimeError(f"Client {client_id} missing 'set_previous_gradient'.")
            print(f"Option B: Switching clients to return_type '{self._client_return_type}'")
            for client in self.clients: client.set_return_type(self._client_return_type)

        # Compute w^1 = w^0 - eta * u^0 (Line 3)
        with torch.no_grad():
            weights_w1 = {name: initial_weights[name] - self.server_lr * u0.get(name, torch.zeros_like(initial_weights[name]))
                          for name in initial_weights}
        self.set_model_weights(weights_w1) # Server now holds w^1
        self.current_server_iteration = 0 # t=0 corresponds to state w^1

        # Initial evaluation and log (only if interval requires it)
        if self.eval_interval > 0 and self.current_server_iteration % self.eval_interval == 0:
            self.log_results_conceptual(avg_train_loss=np.nan) # No train loss yet

        print("Initialization complete. Sending w^1 and starting main loop...")
        # Schedule first actual work for all clients
        active_clients_in_queue = 0
        for client_id in range(self.num_total_clients):
             self.clients[client_id].set_model_weights(copy.deepcopy(weights_w1)) # Send w^1
             delay = simulate_delay(client_id, **self.delay_config)
             completion_time = self.current_wall_time + delay
             heapq.heappush(self.client_completion_events, (completion_time, client_id, 0))
             active_clients_in_queue += 1

        # --- Main Simulation Loop (Algorithm 1, Lines 4-11) ---
        applied_server_updates = 0
        while applied_server_updates < self.max_server_iterations:
            if not self.client_completion_events or active_clients_in_queue == 0:
                print("Warning: No more client events in queue or no active clients. Stopping.")
                break

            try:
                completion_time, client_id, model_version_trained = heapq.heappop(self.client_completion_events)
                active_clients_in_queue -= 1
            except IndexError:
                 print("Warning: Event queue empty during loop check. Stopping.")
                 break

            self.current_wall_time = completion_time
            client_update, loss, samples = self.clients[client_id].get_update() # Returns CPU tensor

            # Handle failed/empty update
            if not client_update or np.isnan(loss):
                 print(f"Warning: Received empty update or NaN loss from client {client_id}. Reassigning task.")
                 current_weights_for_client = self.get_model_weights()
                 self.clients[client_id].set_model_weights(copy.deepcopy(current_weights_for_client))
                 delay = simulate_delay(client_id, **self.delay_config)
                 next_completion_time = self.current_wall_time + delay
                 heapq.heappush(self.client_completion_events, (next_completion_time, client_id, self.current_server_iteration))
                 active_clients_in_queue += 1
                 continue

            # --- Compute Aggregated Update u^t ---
            ut = None
            try:
                if self.option == 'A':
                    gradient_jt = client_update
                    if self.client_gradient_caches.get(client_id) is None: self.client_gradient_caches[client_id] = {}
                    for name in gradient_jt: self.client_gradient_caches[client_id][name] = gradient_jt[name].cpu().clone()
                    ut = self._aggregate_option_a()
                else: # Option B
                    gradient_delta_jt = client_update
                    ut = self._update_option_b(gradient_delta_jt)
            except Exception as e:
                 print(f"!!! Error during server update calculation (Option {self.option}): {e}")
                 traceback.print_exc()
                 # Reassign client work without updating model
                 current_weights_for_client = self.get_model_weights()
                 self.clients[client_id].set_model_weights(copy.deepcopy(current_weights_for_client))
                 delay = simulate_delay(client_id, **self.delay_config)
                 next_completion_time = self.current_wall_time + delay
                 heapq.heappush(self.client_completion_events, (next_completion_time, client_id, self.current_server_iteration))
                 active_clients_in_queue += 1
                 continue

            # --- Apply Global Update ---
            current_weights_t = self.get_model_weights()
            with torch.no_grad():
                 new_weights_tplus1 = {}
                 for name in current_weights_t:
                      if name in ut: new_weights_tplus1[name] = current_weights_t[name] - self.server_lr * ut[name]
                      else: new_weights_tplus1[name] = current_weights_t[name]
            self.set_model_weights(new_weights_tplus1)

            applied_server_updates += 1
            self.current_server_iteration = applied_server_updates

            # --- Send updated model and schedule next task ---
            self.clients[client_id].set_model_weights(copy.deepcopy(new_weights_tplus1))
            delay = simulate_delay(client_id, **self.delay_config)
            next_completion_time = self.current_wall_time + delay
            heapq.heappush(self.client_completion_events, (next_completion_time, client_id, self.current_server_iteration))
            active_clients_in_queue += 1

            # --- Logging and Evaluation ---
            log_loss = loss if not np.isnan(loss) else np.nan
            print(f"Iter: {self.current_server_iteration:<4} | Client {client_id: <3} fin. | SimTime: {self.current_wall_time:>8.2f}s | Loss: {log_loss: >7.4f}")

            if self.current_server_iteration % self.eval_interval == 0 or self.current_server_iteration == self.max_server_iterations:
                 self.log_results_conceptual(avg_train_loss=log_loss)

            # Optional: Check real wall clock time limit
            max_real_time = self.config.get('max_wall_time_seconds', float('inf'))
            if time.time() - self.start_time > max_real_time:
                 print(f"Max real wall clock time ({max_real_time}s) reached.")
                 break

        # --- Finalization ---
        print(f"\n--- StaleSGD Conceptual (Option {self.option}) Simulation Finished ---")
        if applied_server_updates < self.max_server_iterations:
             print(f"Stopped early after {applied_server_updates} server iterations.")
        if not self.results['server_iteration'] or self.results['server_iteration'][-1] != self.current_server_iteration:
             self.log_results_conceptual(avg_train_loss=np.nan)

        return self.results


    # def log_results_conceptual(self, avg_train_loss: Optional[float] = None):
    #      """ Logs evaluation results and timing. """
    #      self.model.to(self.device)
    #      try:
    #           test_loss, test_acc = self.evaluate()
    #      except Exception as e:
    #           print(f"!!! Error during evaluation at iteration {self.current_server_iteration}: {e}")
    #           test_loss, test_acc = float('nan'), float('nan')

    #      current_real_time = time.time() - self.start_time
    #      iter_num = self.current_server_iteration

    #      self.results['server_iteration'].append(iter_num)
    #      self.results['wall_clock_time'].append(current_real_time)
    #      self.results['test_loss'].append(test_loss)
    #      self.results['test_accuracy'].append(test_acc)

    #      # Use last valid train_loss if current is None/NaN
    #      last_train_loss = avg_train_loss if avg_train_loss is not None and not np.isnan(avg_train_loss) else \
    #                        (self.results['train_loss'][-1] if self.results['train_loss'] and not np.isnan(self.results['train_loss'][-1]) else np.nan)
    #      self.results['train_loss'].append(last_train_loss)

    #      # Format for printing
    #      log_train_loss_str = f"{last_train_loss:7.4f}" if not np.isnan(last_train_loss) else "  N/A  "
    #      log_test_loss_str = f"{test_loss:7.4f}" if not np.isnan(test_loss) else "  N/A  "
    #      log_test_acc_str = f"{test_acc:6.2f}%" if not np.isnan(test_acc) else " N/A % "

    #      print(f"EVAL :: Iter: {iter_num:<4} | Real Time: {current_real_time:7.2f}s | Sim Time: {self.current_wall_time:>8.2f}s | Test Loss: {log_test_loss_str} | Test Acc: {log_test_acc_str} | Train Loss: {log_train_loss_str}")
