# server/base_server.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional, Tuple
import time
import copy
# Assuming BaseClient is defined in clients.base_client
# from clients.base_client import BaseClient 

class BaseServer:
    """
    Base class for the central server in Federated Learning.
    Manages the global model, coordinates clients, aggregates updates, 
    and evaluates performance.
    """
    def __init__(self,
                 model: nn.Module,
                 clients: List[Any], # Should ideally be List[BaseClient]
                 test_loader: DataLoader,
                 config: Dict[str, Any],
                 device: Optional[torch.device] = None):
        """
        Initializes the BaseServer.

        Args:
            model (nn.Module): The global model structure.
            clients (List[BaseClient]): A list of client objects participating.
            test_loader (DataLoader): DataLoader for the global test set.
            config (Dict[str, Any]): Configuration dictionary containing parameters like 
                                     max_iterations, eval_interval, etc.
            device (Optional[torch.device]): Device to run the server model on.
        """
        self.model = model
        self.clients = clients
        self.test_loader = test_loader
        self.config = config
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model.to(self.device)
        
        self.current_server_iteration = 0
        self.max_server_iterations = config.get('max_server_iterations', 100) # Default value
        self.eval_interval = config.get('eval_interval', 10) # Evaluate every 10 iterations
        
        self.results: Dict[str, List] = {
            'server_iteration': [],
            'wall_clock_time': [],
            'test_loss': [],
            'test_accuracy': [],
            'train_loss': [] # Average train loss from participating clients
        }
        self.start_time = time.time()

    def get_model_weights(self) -> Dict[str, torch.Tensor]:
        """Returns the state dictionary of the current global model."""
        return self.model.state_dict()

    def set_model_weights(self, weights: Dict[str, torch.Tensor]):
        """Sets the global model weights."""
        self.model.load_state_dict(weights)

    def aggregate(self, updates: List[Tuple[Dict[str, torch.Tensor], int]], **kwargs) -> Dict[str, torch.Tensor]:
        """
        Aggregates updates received from clients. 
        Base implementation performs simple FedAvg on model weights.
        Assumes 'updates' contains tuples of (client_state_dict, num_samples).

        Args:
            updates (List[Tuple[Dict[str, torch.Tensor], int]]): List of updates from clients. 
                                       Each update is a tuple containing the client's 
                                       model state_dict and the number of samples used.
            **kwargs: Additional arguments specific to aggregation strategy.


        Returns:
            Dict[str, torch.Tensor]: The aggregated model weights (state_dict).
        """
        if not updates:
            return self.get_model_weights() # Return current weights if no updates

        total_samples = sum(num_samples for _, num_samples in updates)
        if total_samples == 0:
             return self.get_model_weights() # Avoid division by zero

        aggregated_weights = copy.deepcopy(updates[0][0]) # Start with the first update's structure

        # Zero out the aggregated weights
        for key in aggregated_weights:
            aggregated_weights[key] = torch.zeros_like(aggregated_weights[key])

        # Perform weighted averaging
        for client_weights, num_samples in updates:
            weight = num_samples / total_samples
            for key in aggregated_weights:
                # Ensure weights are on the same device before aggregation
                aggregated_weights[key] = aggregated_weights[key].float() + client_weights[key].to(self.device) * weight
                 
        return aggregated_weights

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluates the current global model on the test dataset.

        Returns:
            Tuple[float, float]: Average test loss and accuracy.
        """
        self.model.eval() # Set model to evaluation mode
        test_loss = 0.0
        correct = 0
        total = 0
        loss_fn = nn.CrossEntropyLoss(reduction='sum') # Sum loss for averaging later

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += loss_fn(output, target).item() 
                _, predicted = torch.max(output.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()

        avg_loss = test_loss / total if total > 0 else 0.0
        accuracy = 100. * correct / total if total > 0 else 0.0
        
        # print(f"Evaluation - Iteration: {self.current_server_iteration}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        return avg_loss, accuracy
        
    def log_results(self, avg_train_loss: Optional[float] = None):
         """Logs evaluation results and timing."""
         test_loss, test_acc = self.evaluate()
         current_time = time.time() - self.start_time
         
         self.results['server_iteration'].append(self.current_server_iteration)
         self.results['wall_clock_time'].append(current_time)
         self.results['test_loss'].append(test_loss)
         self.results['test_accuracy'].append(test_acc)
         if avg_train_loss is not None:
              self.results['train_loss'].append(avg_train_loss)
         else:
             # Append None or last value if no training happened
             self.results['train_loss'].append(self.results['train_loss'][-1] if self.results['train_loss'] else None)


         print(f"Iter: {self.current_server_iteration:<4} | Time: {current_time:7.2f}s | Test Loss: {test_loss:7.4f} | Test Acc: {test_acc:6.2f}% | Avg Train Loss: {f'{avg_train_loss:7.4f}' if avg_train_loss is not None else 'N/A'}")


    def run(self):
        """
        Runs the federated learning simulation.
        This base implementation needs to be significantly extended or overridden 
        by specific algorithm subclasses (synchronous vs asynchronous logic).
        """
        raise NotImplementedError("The 'run' method must be implemented by subclasses.")
        
        # --- Conceptual Structure (to be implemented by subclasses) ---
        # self.start_time = time.time()
        # for t in range(self.max_server_iterations):
        #     self.current_server_iteration = t
        #     
        #     # 1. Select/Activate Clients (depends on sync/async)
        #     selected_clients = self.select_clients(t) 
        #     
        #     # 2. Broadcast Model (sync or async trigger)
        #     current_weights = self.get_model_weights()
        #     # ... logic to send weights to selected_clients ...
        #     
        #     # 3. Client Training & Update Collection (sync or async handling)
        #     #    - Sync: Wait for all selected clients
        #     #    - Async: Handle updates as they arrive, potentially managing a buffer
        #     updates = self.collect_updates(selected_clients, current_weights) # Placeholder
        #     
        #     # 4. Aggregate Updates
        #     aggregated_weights = self.aggregate(updates)
        #     
        #     # 5. Update Global Model
        #     self.set_model_weights(aggregated_weights)
        #     
        #     # 6. Evaluate and Log
        #     if t % self.eval_interval == 0 or t == self.max_server_iterations - 1:
        #         avg_train_loss = self.calculate_avg_train_loss(updates) # Placeholder
        #         self.log_results(avg_train_loss)
                
        #     self.current_server_iteration += 1 # Increment server iteration concept

# Example Usage (Illustrative - `run` needs implementation in subclasses)
if __name__ == '__main__':
     # Need dummy model, clients, loader, config
     dummy_model = nn.Linear(10, 2)
     dummy_loader = DataLoader(torch.randn(100, 10), batch_size=10) # Dummy test loader
     
     class MockClient: # Mock client for demonstration
         def __init__(self, id):
             self.client_id = id
         def set_model_weights(self, weights): pass
         def get_update(self): return ({k: v.clone() for k,v in dummy_model.state_dict().items()}, 0.5, 10) # Dummy update

     dummy_clients = [MockClient(i) for i in range(5)]
     dummy_config = {'max_server_iterations': 10, 'eval_interval': 1}

     server = BaseServer(dummy_model, dummy_clients, dummy_loader, dummy_config)
     print("BaseServer initialized.")
     
     # Evaluate initial model
     server.evaluate()
     
     # Example of aggregation (assuming some updates arrived)
     client_updates = [client.get_update() for client in dummy_clients[:2]] # Get updates from first 2 clients
     new_weights = server.aggregate([(up[0], up[2]) for up in client_updates]) # Aggregate state_dicts
     server.set_model_weights(new_weights)
     print("\nAggregated weights from 2 clients and updated server model.")
     
     # Evaluate after aggregation
     server.evaluate()
     
     # server.run() # Would raise NotImplementedError