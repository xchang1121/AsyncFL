# experiments/experiment_manager.py

import torch
import torch.nn as nn
import yaml
import os
import copy
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Type, Tuple

# Import necessary components from other modules
from datasets.cifar10 import CIFAR10Dataset
# from datasets.femnist import FEMNISTDataset # Add when implemented
from models.resnet import ResNet18_CIFAR
from models.cnn import CNN_CIFAR
from clients.base_client import BaseClient
from server.base_server import BaseServer 

# Import Server implementations
from algorithms.fedavg import FedAvgServer
from algorithms.fedbuff import FedBuffServer
from algorithms.ca2fl import CA2FLServer
from algorithms.vanilla_asgd import VanillaASGDServer
from algorithms.delay_adaptive import DelayAdaptiveAFLServer
from algorithms.malenia_sgd import MaleniaSGDServer
from algorithms.stalesgd import StaleSGDServer
from algorithms.stalesgd_conceptual import StaleSGDConceptualServer

# Import utilities
from utils.simulation import assign_client_speeds

# Map algorithm names from config to Server classes
ALGORITHM_MAP: Dict[str, Type[BaseServer]] = {
    "fedavg": FedAvgServer,
    "fedbuff": FedBuffServer,
    "ca2fl": CA2FLServer,
    "vanilla_asgd": VanillaASGDServer,
    "delay_adaptive_afl": DelayAdaptiveAFLServer,
    "malenia_sgd": MaleniaSGDServer,
    "stalesgd_bda": StaleSGDServer,
    "stalesgd_conceptual": StaleSGDConceptualServer
}

# Map dataset names from config to Dataset classes
DATASET_MAP = {
    "cifar10": CIFAR10Dataset,
    # "femnist": FEMNISTDataset, 
}

# Map model names from config to model functions/classes
MODEL_MAP = {
    "resnet18_cifar": ResNet18_CIFAR,
    "cnn_cifar": CNN_CIFAR,
}

class ExperimentManager:
    """ Manages the setup and execution of federated learning experiments. """

    def __init__(self, config_path: str):
        """
        Loads configuration and sets up the manager.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.base_seed = self.config.get('seed', 42)
        self.repetitions = self.config.get('repetitions', 1)
        
        # Create results directory
        self.results_base_dir = self.config.get('results_dir', './results')
        self.experiment_name = self.config.get('experiment_name', 'fl_experiment')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_run_dir = os.path.join(self.results_base_dir, f"{self.experiment_name}_{timestamp}")
        os.makedirs(self.experiment_run_dir, exist_ok=True)
        
        # Save config used for this run
        with open(os.path.join(self.experiment_run_dir, 'config.yaml'), 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")


    def _set_seed(self, seed):
        """ Sets random seeds for reproducibility. """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            # Potentially set deterministic algorithms
            # torch.backends.cudnn.deterministic = True
            # torch.backends.cudnn.benchmark = False

    def _setup_single_run(self, run_seed: int) -> Tuple[Any, List[BaseClient], Any, Dict[str, Any]]:
        """ Sets up dataset, model, and clients for a single run. """
        self._set_seed(run_seed)
        
        # --- Dataset Setup ---
        ds_config = self.config['dataset']
        dataset_class = DATASET_MAP.get(ds_config['name'].lower())
        if dataset_class is None:
            raise ValueError(f"Unknown dataset: {ds_config['name']}")
        
        dataset = dataset_class(data_root=ds_config['data_root'])
        dataset.load_data()
        
        num_clients = ds_config['partition']['num_clients']
        # Partition data - specific logic might depend on dataset type
        if ds_config['name'].lower() == 'cifar10':
             if ds_config['partition']['type'] == 'dirichlet':
                 dataset.partition_data(num_clients=num_clients, 
                                       alpha=ds_config['partition']['alpha'], 
                                       seed=run_seed)
             else:
                  # Add other partitioning methods if needed (e.g., IID)
                  raise NotImplementedError(f"Partition type '{ds_config['partition']['type']}' not implemented for CIFAR-10.")
        # elif ds_config['name'].lower() == 'femnist':
        #      # FEMNIST uses its native partitioning based on users
        #      dataset.partition_data(num_clients=num_clients) # Adjust as per FEMNIST class implementation
        else:
            raise ValueError(f"Partitioning logic not defined for dataset: {ds_config['name']}")

        test_loader = dataset.get_test_dataloader(batch_size=self.config['simulation']['batch_size'] * 2) # Often use larger batch for testing

        # --- Model Setup ---
        model_config = self.config['model']
        model_builder = MODEL_MAP.get(model_config['name'].lower())
        if model_builder is None:
             raise ValueError(f"Unknown model: {model_config['name']}")
        # Pass num_classes from dataset to model builder
        model = model_builder(num_classes=dataset.num_classes)
        model.to(self.device)

        # --- Client Setup ---
        clients = []
        sim_config = self.config['simulation']
        for i in range(num_clients):
            client_dataloader = dataset.get_train_dataloader(client_id=i, 
                                                             batch_size=sim_config['batch_size'])
            # Need to create a *new* model instance for each client, 
            # otherwise they share weights unintentionally during simulation.
            client_model = copy.deepcopy(model) 
            client = BaseClient(client_id=i,
                                model=client_model, # Pass model structure
                                dataloader=client_dataloader,
                                optimizer_name=sim_config['optimizer']['name'],
                                optimizer_params=sim_config['optimizer']['params'],
                                loss_fn=nn.CrossEntropyLoss(), # Or get from config
                                local_epochs=sim_config['local_epochs'],
                                device=self.device)
            clients.append(client)
            
        # Assign client speeds if mixed mode delay is used
        delay_type = sim_config.get('delay_config', {}).get('distribution_type', '')
        if 'mixed' in delay_type:
            slow_fraction = sim_config.get('delay_config', {}).get('params', {}).get('slow_fraction', 0.2)
            assign_client_speeds(list(range(num_clients)), slow_fraction=slow_fraction)


        return model, clients, test_loader, sim_config


    def run_all_experiments(self):
        """ Runs the experiments for all algorithms and repetitions defined in the config. """
        
        all_results = {} # Store results per algorithm per run

        for i in range(self.repetitions):
            run_seed = self.base_seed + i
            print(f"\n===== Starting Repetition {i+1}/{self.repetitions} (Seed: {run_seed}) =====")
            
            # Setup dataset, initial model, clients for this repetition
            initial_model_structure, clients, test_loader, sim_config = self._setup_single_run(run_seed)
            
            for algo_config in self.config['algorithms']:
                algo_name = algo_config['name']
                algo_params = algo_config['params']
                print(f"\n----- Running Algorithm: {algo_name} -----")
                
                # Get the correct server class
                server_class = ALGORITHM_MAP.get(algo_name.lower())
                if server_class is None:
                    print(f"Warning: Unknown algorithm '{algo_name}'. Skipping.")
                    continue
                    
                # Create a fresh copy of the initial model for each algorithm run
                current_run_model = copy.deepcopy(initial_model_structure)
                current_run_model.to(self.device)
                
                # Combine general simulation config with algorithm-specific params
                current_config = {**sim_config, **algo_params, 
                                  'max_server_iterations': sim_config.get('max_server_iterations'),
                                  'max_applied_updates': sim_config.get('max_applied_updates'),
                                  'max_wall_time_seconds': sim_config.get('max_wall_time_seconds'),
                                  'eval_interval': sim_config.get('eval_interval')}


                # Instantiate the server
                server = server_class(model=current_run_model, 
                                      clients=clients,         # Clients are reset by server before training
                                      test_loader=test_loader,
                                      config=current_config,
                                      device=self.device)

                # Run the simulation
                results = server.run() # Server's run method handles the loop and logging

                # Store results
                if algo_name not in all_results:
                    all_results[algo_name] = []
                all_results[algo_name].append(results)
                
                # Save results for this run
                results_filename = f"{algo_name}_seed{run_seed}_results.pt"
                results_path = os.path.join(self.experiment_run_dir, results_filename)
                torch.save(results, results_path)
                print(f"Results saved to {results_path}")

        print("\n===== All Experiment Repetitions Finished =====")
        return all_results


# Example usage (called from main.py)
# if __name__ == '__main__':
#     config_file = 'config.yaml' # Or get from argparse
#     manager = ExperimentManager(config_path=config_file)
#     final_results = manager.run_all_experiments()
#     # Post-processing or plotting could happen here
#     print("\nExperiment manager finished.")
