# experiments/experiment_manager.py


import yaml
import os
import copy
import random
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Type, Tuple
import traceback # For better error printing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# --- Import necessary components ---

# Dataset Imports
from datasets.cifar10 import CIFAR10Dataset
# from datasets.femnist import FEMNISTDataset # Add when implemented
from datasets.base_dataset import BaseDataset # Useful for type hinting

# Model Imports
from models.resnet import ResNet18_CIFAR
from models.cnn import CNN_CIFAR

# Client Import (Use the modified version that supports all return types)
from clients.base_client import BaseClient

# Server Imports
from server.base_server import BaseServer
from algorithms.fedavg import FedAvgServer
from algorithms.fedbuff import FedBuffServer
from algorithms.ca2fl import CA2FLServer
from algorithms.vanilla_asgd import VanillaASGDServer
from algorithms.delay_adaptive import DelayAdaptiveAFLServer
from algorithms.malenia_sgd import MaleniaSGDServer
from algorithms.stalesgd import StaleSGDServer # BDA Variant
# Import the new conceptual server
from algorithms.stalesgd_conceptual import StaleSGDConceptualServer

# Utility Imports
from utils.simulation import assign_client_speeds # Only needed if using mixed delays

# --- Mappings ---

# Map algorithm names from config to Server classes
# Ensure this matches the server class names exactly
ALGORITHM_MAP: Dict[str, Type[BaseServer]] = {
    "fedavg": FedAvgServer,
    "fedbuff": FedBuffServer,
    "ca2fl": CA2FLServer,
    "vanilla_asgd": VanillaASGDServer,
    "delay_adaptive_afl": DelayAdaptiveAFLServer,
    "malenia_sgd": MaleniaSGDServer,
    "stalesgd_bda": StaleSGDServer, # The practical BDA variant
    "stalesgd_conceptual": StaleSGDConceptualServer, # The conceptual variant (handles A/B internally)
}

# Map dataset names from config to Dataset classes
DATASET_MAP: Dict[str, Type[BaseDataset]] = {
    "cifar10": CIFAR10Dataset,
    # "femnist": FEMNISTDataset,
}

# Map model names from config to model functions/classes
MODEL_MAP = {
    "resnet18_cifar": ResNet18_CIFAR,
    "cnn_cifar": CNN_CIFAR,
}

# --- Experiment Manager Class ---

class ExperimentManager:
    """ Manages the setup and execution of federated learning experiments. """

    def __init__(self, config_path: str):
        """
        Loads configuration and sets up the manager.

        Args:
            config_path (str): Path to the YAML configuration file.
        """
        if not os.path.exists(config_path):
             raise FileNotFoundError(f"Configuration file not found at: {config_path}")

        print(f"Loading configuration from: {config_path}")
        with open(config_path, 'r') as f:
            try:
                self.config = yaml.safe_load(f)
            except yaml.YAMLError as e:
                 print(f"Error parsing YAML file: {e}")
                 raise

        # Basic configuration validation
        if not isinstance(self.config, dict):
             raise ValueError("Invalid configuration format. Root should be a dictionary.")
        required_keys = ['dataset', 'model', 'simulation', 'algorithms']
        for key in required_keys:
             if key not in self.config:
                  raise ValueError(f"Missing required configuration section: '{key}'")


        self.base_seed = self.config.get('seed', 42)
        self.repetitions = self.config.get('repetitions', 1)

        # Create results directory
        self.results_base_dir = self.config.get('results_dir', './results')
        self.experiment_name = self.config.get('experiment_name', 'fl_experiment')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Extract algorithm name if exactly one algorithm is listed
        algorithms = self.config.get('algorithms', [])
        if isinstance(algorithms, list) and len(algorithms) == 1:
            algo_name = algorithms[0].get('name', 'unknown_algo')
            run_name = f"{timestamp}_{self.experiment_name}_{algo_name}"
        else:
            run_name = f"{timestamp}_{self.experiment_name}"

        self.experiment_run_dir = os.path.join(self.results_base_dir, run_name)
        try:
            os.makedirs(self.experiment_run_dir, exist_ok=True)
            print(f"Results will be saved in: {self.experiment_run_dir}")
        except OSError as e:
             print(f"Error creating results directory {self.experiment_run_dir}: {e}")
             raise

        # Save config used for this run
        try:
            config_save_path = os.path.join(self.experiment_run_dir, 'config.yaml')
            with open(config_save_path, 'w') as f:
                yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
        except IOError as e:
             print(f"Warning: Could not save config file to results directory: {e}")

        # Determine device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        if str(self.device) == "cuda":
             if torch.cuda.is_available() and torch.cuda.device_count() > 0:
                  try:
                    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
                  except Exception as e:
                       print(f"Could not get CUDA device name: {e}")
             else:
                  print("CUDA selected, but no CUDA devices found by PyTorch.")


    def _set_seed(self, seed):
        """ Sets random seeds for reproducibility. """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        print(f"Set seed for reproducibility: {seed}")


    def _setup_single_run(self, run_seed: int) -> Tuple[nn.Module, List[BaseClient], DataLoader, Dict[str, Any]]:
        """ Sets up dataset, model, and clients for a single run. Returns CPU model structure. """
        self._set_seed(run_seed)

        # --- Dataset Setup ---
        ds_config = self.config['dataset']
        ds_name = ds_config.get('name', '').lower()
        dataset_class = DATASET_MAP.get(ds_name)
        if dataset_class is None:
            raise ValueError(f"Unknown dataset specified in config: {ds_config.get('name')}")

        print(f"Setting up dataset: {ds_name}")
        dataset = dataset_class(data_root=ds_config.get('data_root', './data'))
        dataset.load_data() # Downloads if necessary

        partition_config = ds_config.get('partition', {})
        num_clients = partition_config.get('num_clients')
        if num_clients is None or num_clients <= 0:
            raise ValueError("Invalid or missing 'num_clients' in dataset partition config.")

        partition_type = partition_config.get('type', '').lower()
        print(f"Partitioning data for {num_clients} clients using type: '{partition_type}'")
        if partition_type == 'dirichlet':
            alpha = partition_config.get('alpha')
            if alpha is None or alpha <= 0:
                 raise ValueError("Dirichlet partitioning requires a positive 'alpha' parameter.")
            dataset.partition_data(num_clients=num_clients, alpha=alpha, seed=run_seed)
        # Add other partitioning types here if needed
        else:
            raise NotImplementedError(f"Partition type '{partition_type}' not implemented or invalid for dataset '{ds_name}'.")

        sim_config = self.config['simulation']
        test_batch_size = sim_config.get('test_batch_size', sim_config.get('batch_size', 64) * 2)
        test_loader = dataset.get_test_dataloader(batch_size=test_batch_size)
        print(f"Test DataLoader created with batch size: {test_batch_size}")

        # --- Model Setup ---
        model_config = self.config['model']
        model_name = model_config.get('name', '').lower()
        model_builder = MODEL_MAP.get(model_name)
        if model_builder is None:
             raise ValueError(f"Unknown model specified in config: {model_config.get('name')}")

        num_classes = getattr(dataset, 'num_classes', model_config.get('num_classes'))
        if num_classes is None:
             raise ValueError("Could not determine 'num_classes' for the model from dataset or config.")

        print(f"Creating model: {model_name} with {num_classes} classes.")
        # Create model structure on CPU first
        model_structure = model_builder(num_classes=num_classes)

        # --- Client Setup ---
        print("Setting up clients...")
        clients = []
        client_batch_size = sim_config.get('batch_size', 64)
        local_epochs = sim_config.get('local_epochs', 1)
        optimizer_config = sim_config.get('optimizer', {})
        opt_name = optimizer_config.get('name', 'sgd')
        opt_params = optimizer_config.get('params', {'lr': 0.01})

        # Ensure BaseClient uses the modified version by checking for set_return_type
        if not hasattr(BaseClient(0, model_structure, DataLoader([]), device=torch.device('cpu')), 'set_return_type'):
             raise RuntimeError("The BaseClient class being used is outdated. Please ensure clients/base_client.py has been updated.")

        for i in range(num_clients):
            try:
                # Use shuffle=True for training data by default
                client_dataloader = dataset.get_train_dataloader(client_id=i, batch_size=client_batch_size, shuffle=True)
            except ValueError as e:
                 print(f"Error getting dataloader for client {i}: {e}. Skipping client.")
                 continue
            except Exception as e:
                 print(f"Unexpected error getting dataloader for client {i}: {e}. Skipping client.")
                 continue

            # Create a fresh model instance for each client from the structure
            client_model = copy.deepcopy(model_structure)
            try:
                 # Initialize with a default return_type; it will be set correctly later per algorithm
                 client = BaseClient(client_id=i,
                                     model=client_model,
                                     dataloader=client_dataloader,
                                     optimizer_name=opt_name,
                                     optimizer_params=opt_params,
                                     loss_fn=nn.CrossEntropyLoss(),
                                     local_epochs=local_epochs,
                                     device=self.device, # Pass the target device
                                     return_type='weights') # Default init type
                 clients.append(client)
            except Exception as e:
                 print(f"Error creating BaseClient instance for client {i}: {e}. Skipping client.")
                 traceback.print_exc()

        if not clients:
             raise RuntimeError("Failed to create any clients. Check dataset partitioning and client setup.")
        print(f"Successfully created {len(clients)} clients.")

        # Assign client speeds if mixed mode delay is used
        delay_config = sim_config.get('delay_config', {})
        delay_type = delay_config.get('distribution_type', '')
        if 'mixed' in delay_type:
            print("Assigning client speeds for mixed delay mode...")
            slow_fraction = delay_config.get('params', {}).get('slow_fraction', 0.2)
            if not 0 <= slow_fraction <= 1:
                 print(f"Warning: Invalid slow_fraction ({slow_fraction}). Using default 0.2.")
                 slow_fraction = 0.2
            try:
                 assign_client_speeds([c.client_id for c in clients], slow_fraction=slow_fraction)
            except Exception as e:
                 print(f"Error during client speed assignment: {e}")

        # Return model structure on CPU, clients list, test loader, and sim config
        return model_structure.cpu(), clients, test_loader, sim_config


    def run_all_experiments(self):
        """ Runs the experiments for all algorithms and repetitions defined in the config. """

        all_run_results = {} # Store results {algo_name: [rep1_results, rep2_results, ...]}

        for i in range(self.repetitions):
            run_seed = self.base_seed + i
            print(f"\n{'='*25} Starting Repetition {i+1}/{self.repetitions} (Seed: {run_seed}) {'='*25}")

            try:
                # Setup dataset, initial model structure (CPU), clients list, test loader
                initial_model_structure_cpu, clients, test_loader, sim_config = self._setup_single_run(run_seed)
            except Exception as e:
                print(f"\n!!!!!! Error during setup for Repetition {i+1} (Seed: {run_seed}) !!!!!!")
                print(f"Error: {e}")
                traceback.print_exc()
                print("!!!!!! Skipping this repetition !!!!!!")
                continue # Skip to the next repetition


            # --- Loop through each algorithm configuration ---
            for algo_config in self.config.get('algorithms', []):
                if not isinstance(algo_config, dict) or 'name' not in algo_config:
                     print(f"Warning: Invalid algorithm configuration format found: {algo_config}. Skipping.")
                     continue

                algo_name = algo_config['name']
                algo_params = algo_config.get('params', {})
                print(f"\n----- Running Algorithm: {algo_name} -----")

                server_class = ALGORITHM_MAP.get(algo_name.lower())
                if server_class is None:
                    print(f"Warning: Unknown algorithm '{algo_name}' defined in config. Skipping.")
                    continue

                # Create a fresh model instance for this algorithm run and move to device
                current_run_model = copy.deepcopy(initial_model_structure_cpu).to(self.device)

                # Combine general simulation config with algorithm-specific params
                current_config = {
                    **sim_config,
                    **algo_params,
                    'num_clients': len(clients),
                    'max_server_iterations': sim_config.get('max_server_iterations'),
                    'max_applied_updates': sim_config.get('max_applied_updates'),
                    'max_wall_time_seconds': sim_config.get('max_wall_time_seconds'),
                    'eval_interval': sim_config.get('eval_interval')
                 }


                # --- Configure Clients for the Current Algorithm ---
                required_type = 'weights' # Default assumption
                algo_name_lower = algo_name.lower()

                try:
                    if algo_name_lower == 'fedavg':
                        required_type = 'weights'
                    elif algo_name_lower in ['fedbuff', 'ca2fl', 'vanilla_asgd', 'delay_adaptive_afl']:
                        required_type = 'weights' # These servers currently calculate delta internally
                    elif algo_name_lower in ['malenia_sgd', 'stalesgd_bda', 'stalesgd_conceptual']:
                        # StaleSGD Conceptual (Option A/B handled internally) needs gradient or delta
                        # The StaleSGDConceptualServer class itself checks the 'conceptual_stalegsgd_option'
                        # param and sets the appropriate internal _client_return_type.
                        # Here, we need to determine the type needed *by the server* for its initial setup
                        # and potentially subsequent steps. Let's base it on the config if possible.
                        if algo_name_lower == 'stalesgd_conceptual':
                            option = current_config.get('conceptual_stalegsgd_option', 'A').upper()
                            required_type = 'gradient_delta' if option == 'B' else 'gradient'
                        else: # malenia_sgd, stalesgd_bda need gradient
                            required_type = 'gradient'
                    else:
                        print(f"Warning: Unknown algorithm '{algo_name}'. Defaulting client return_type to '{required_type}'.")

                    print(f"Configuring {len(clients)} clients for '{algo_name}' with return_type='{required_type}'...")
                    for client in clients:
                        if isinstance(client, BaseClient):
                            client.set_return_type(required_type) # Use the setter
                        # else: Handled during setup
                except Exception as e:
                     print(f"!!! Error configuring clients for {algo_name}: {e}. Skipping algorithm.")
                     traceback.print_exc()
                     continue # Skip to the next algorithm


                # --- Instantiate and Run the Server ---
                server = None
                try:
                    server = server_class(model=current_run_model,
                                          clients=clients,
                                          test_loader=test_loader,
                                          config=current_config,
                                          device=self.device)
                except Exception as e:
                     print(f"!!! Error instantiating server for {algo_name}: {e}")
                     traceback.print_exc()
                     continue # Skip to next algorithm

                try:
                    print(f"Starting server run for {algo_name}...")
                    results = server.run()
                    print(f"Finished server run for {algo_name}.")
                except Exception as e:
                    print(f"!!! Error during server run for {algo_name}: {e}")
                    traceback.print_exc()
                    results = None

                # --- Store and Save Results ---
                if results is not None and isinstance(results, dict):
                    if algo_name not in all_run_results:
                        all_run_results[algo_name] = []
                    all_run_results[algo_name].append(results)

                    results_filename = f"{algo_name}_seed{run_seed}_results.pt"
                    results_path = os.path.join(self.experiment_run_dir, results_filename)
                    try:
                        torch.save(results, results_path)
                        print(f"Results for {algo_name} (Seed {run_seed}) saved to {results_path}")
                    except Exception as e:
                         print(f"!!! Error saving results to {results_path}: {e}")
                else:
                     print(f"Skipping result storage for {algo_name} due to run error or invalid result format.")


            # --- End of loop for algorithms ---
        # --- End of loop for repetitions ---

        print(f"\n{'='*25} All Experiment Repetitions Finished {'='*25}")

        # --- Final Save (Combined results across repetitions) ---
        combined_results_path = os.path.join(self.experiment_run_dir, "all_repetitions_results.pt")
        try:
             torch.save(all_run_results, combined_results_path)
             print(f"Combined results across all repetitions saved to: {combined_results_path}")
        except Exception as e:
             print(f"!!! Error saving combined results: {e}")


        return all_run_results
