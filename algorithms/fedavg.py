# algorithms/fedavg.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Tuple
import random
import time
import copy

# Assuming BaseServer is in server.base_server and BaseClient in clients.base_client
from server.base_server import BaseServer
# from clients.base_client import BaseClient # Import if type hinting is desired

class FedAvgServer(BaseServer):
    """
    Implements the synchronous Federated Averaging (FedAvg) algorithm.
    Inherits from BaseServer and overrides the 'run' method.
    """
    def __init__(self,
                 model: nn.Module,
                 clients: List[Any], # List[BaseClient]
                 test_loader: DataLoader,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initializes the FedAvgServer.

        Args:
            model (nn.Module): The global model structure.
            clients (List[BaseClient]): A list of client objects.
            test_loader (DataLoader): DataLoader for the global test set.
            config (Dict[str, Any]): Configuration dictionary. Expected keys include:
                                     'max_server_iterations', 'eval_interval', 
                                     'client_fraction' (fraction of clients per round),
                                     'num_clients' (total number of clients).
            device (Optional[torch.device]): Device to run the server model on.
        """
        super().__init__(model, clients, test_loader, config, device)
        
        self.client_fraction = config.get('client_fraction', 0.1) # Default 10% participation
        if not (0 < self.client_fraction <= 1.0):
             raise ValueError("Client fraction must be between 0 (exclusive) and 1 (inclusive).")
        self.num_clients_per_round = max(1, int(self.client_fraction * len(self.clients)))


    def run(self):
        """
        Runs the synchronous FedAvg simulation loop.
        """
        print(f"Starting FedAvg simulation...")
        print(f"Total clients: {len(self.clients)}")
        print(f"Clients per round: {self.num_clients_per_round} ({self.client_fraction*100:.1f}%)")
        print(f"Max server iterations: {self.max_server_iterations}")
        print(f"Local epochs per client: {self.clients[0].local_epochs if self.clients else 'N/A'}") # Assumes homogeneity for printing
        print(f"Evaluation interval: {self.eval_interval} iterations")
        print(f"Device: {self.device}")
        
        self.start_time = time.time()

        # Initial evaluation
        if self.eval_interval > 0:
             self.log_results(avg_train_loss=None) # Log initial state

        for t in range(self.max_server_iterations):
            self.current_server_iteration = t
            print(f"\n--- Server Iteration {t+1}/{self.max_server_iterations} ---")

            # 1. Select Clients
            selected_client_indices = random.sample(range(len(self.clients)), self.num_clients_per_round)
            selected_clients = [self.clients[i] for i in selected_client_indices]
            print(f"Selected clients: {[c.client_id for c in selected_clients]}")

            # 2. Broadcast Model & Client Training (Synchronous Simulation)
            current_weights = self.get_model_weights()
            # Use deepcopy to avoid clients modifying the same weights object if running truly in parallel
            # For simulation, passing the state_dict might be okay if clients don't modify it in-place
            # but deepcopy is safer conceptually.
            weights_to_send = copy.deepcopy(current_weights) 
            
            client_updates: List[Tuple[Dict[str, torch.Tensor], float, int]] = [] # Store (update, loss, samples)
            
            round_start_time = time.time()
            for client in selected_clients:
                # Send weights
                client.set_model_weights(weights_to_send) 
                
                # Trigger local training
                # In a real sync setting, server waits here. Simulation does it sequentially.
                update_data, loss, samples = client.get_update() 
                client_updates.append((update_data, loss, samples))
                # print(f"Client {client.client_id} finished training.")

            round_duration = time.time() - round_start_time
            print(f"Round {t+1} client training finished in {round_duration:.2f}s (simulated sync).")


            # 3. Aggregate Updates
            # BaseServer.aggregate expects list of (state_dict, num_samples)
            aggregation_input = [(up[0], up[2]) for up in client_updates]
            aggregated_weights = self.aggregate(aggregation_input)
            
            # 4. Update Global Model
            self.set_model_weights(aggregated_weights)
            
            # Calculate average training loss for the round
            total_loss = sum(loss * samples for _, loss, samples in client_updates)
            total_samples = sum(samples for _, _, samples in client_updates)
            avg_train_loss = total_loss / total_samples if total_samples > 0 else 0.0

            # 5. Evaluate and Log
            if (t + 1) % self.eval_interval == 0 or (t + 1) == self.max_server_iterations:
                self.log_results(avg_train_loss=avg_train_loss)
            else:
                 # Optionally print just the training loss
                 current_time = time.time() - self.start_time
                 print(f"Iter: {t:<4} | Time: {current_time:7.2f}s | Avg Train Loss: {avg_train_loss:7.4f}")


        print("\nFedAvg simulation finished.")
        # Return final results
        return self.results

# Example Usage (Illustrative - Needs full setup with actual clients, model, data)
if __name__ == '__main__':
     # Setup similar to BaseServer example
     dummy_model = nn.Linear(10, 2)
     dummy_loader = DataLoader(torch.randn(100, 10), batch_size=10)
     
     class MockClient:
         def __init__(self, id):
             self.client_id = id
             self.model = copy.deepcopy(dummy_model) # Each client gets a model copy
             self.device = torch.device('cpu')
             self.model.to(self.device)
             self.local_epochs = 1 # For printing example
             self.dataloader = DataLoader(torch.randn(20,10), batch_size=5) # Dummy loader
             self.loss_fn = nn.CrossEntropyLoss()
             self.optimizer_params = {'lr': 0.1}
             self.optimizer_name = 'sgd'

         def set_model_weights(self, weights): 
             self.model.load_state_dict(weights)
             
         def get_update(self): 
             # Simulate some training
             self.model.train()
             optimizer = optim.SGD(self.model.parameters(), **self.optimizer_params)
             avg_loss = 0
             count = 0
             for _ in range(self.local_epochs):
                  for data, target in [(torch.randn(5,10), torch.ones(5).long()) for _ in range(4)]: # Dummy batches
                       optimizer.zero_grad()
                       out = self.model(data.to(self.device))
                       loss = self.loss_fn(out, target.to(self.device))
                       loss.backward()
                       optimizer.step()
                       avg_loss += loss.item()
                       count += 1
             return (copy.deepcopy(self.model.state_dict()), avg_loss/count if count>0 else 0, 20) # Return state_dict, loss, samples

     dummy_clients = [MockClient(i) for i in range(10)] # 10 clients total
     dummy_config = {
         'max_server_iterations': 20, 
         'eval_interval': 5,
         'client_fraction': 0.5 # Select 5 clients per round
     }

     server = FedAvgServer(dummy_model, dummy_clients, dummy_loader, dummy_config, device=torch.device('cpu'))
     
     # Run the simulation
     results = server.run()
     
     # Print final results summary (optional)
     print("\n--- Final Results Summary ---")
     for key, values in results.items():
         print(f"{key}: {values}")