# Asynchronous Federated Learning Algorithm Comparison

## Overview

This project implements and simulates various asynchronous federated learning (AFL) algorithms, including AC-AFL and its BDA variant, alongside several synchronous and asynchronous baselines. The goal is to provide a framework for comparing these algorithms under different conditions of data heterogeneity (non-IID) and communication/computation delays, based on the experiments and algorithms described in the referenced research papers.

## Features

* **Multiple Algorithms**: Implements ACAFL, ACAFL-BDA, FedAvg, Vanilla ASGD, FedBuff, CA2FL, Delay-adaptive AFL, and Malenia SGD.
* **Datasets**: Supports CIFAR-10 with plans for FEMNIST.
* **Data Heterogeneity**: Simulates non-IID data distributions using Dirichlet partitioning ($\alpha$ parameter).
* **Delay Simulation**: Models client delays using Exponential or Mixed Mode distributions.
* **Models**: Includes ResNet-18 and a standard CNN adapted for CIFAR-10.
* **Configuration**: Experiments are driven by a YAML configuration file.
* **Logging**: Saves detailed results (accuracy, loss vs. server iteration/applied updates and wall-clock time) for analysis.

## Project Structure

```
stalesgd_afl_comparison/
├── main.py                   # Main script to run experiments
├── config.yaml               # Configuration file for experiments
├── datasets/                 # Data loading and partitioning logic
│   ├── __init__.py
│   ├── base_dataset.py       # Base dataset class
│   ├── cifar10.py            # CIFAR-10 implementation
│   ├── femnist.py            # Placeholder for FEMNIST
│   └── data_utils.py         # Dirichlet partitioning utility
├── models/                   # Model architectures
│   ├── __init__.py
│   ├── resnet.py             # ResNet-18 implementation
│   └── cnn.py                # Standard CNN implementation
├── clients/                  # Client-side logic
│   ├── __init__.py
│   └── base_client.py        # Base client class (local training)
├── server/                   # Server-side logic
│   ├── __init__.py
│   └── base_server.py        # Base server class (coordination, evaluation)
├── algorithms/               # Specific FL/AFL algorithm implementations
│   ├── __init__.py
│   ├── fedavg.py             # FedAvg (Sync)
│   ├── vanilla_asgd.py       # Vanilla ASGD (Async)
│   ├── fedbuff.py            # FedBuff (Async)
│   ├── ca2fl.py              # CA2FL (Async)
│   ├── delay_adaptive.py     # Delay-Adaptive AFL (Async)
│   ├── malenia_sgd.py        # Malenia SGD (Async)
│   └── stalesgd.py           # ACAFL-BDA (Async)
├── experiments/              # Experiment management
│   ├── __init__.py
│   └── experiment_manager.py # Sets up and runs experiments from config
├── utils/                    # Utility functions
│   ├── __init__.py
│   ├── simulation.py         # Delay simulation functions
│   ├── logger.py            # Placeholder for logging utilities
│   └── plotting.py           # Placeholder for plotting utilities
└── requirements.txt          # Python package requirements
```

## Setup

1.  **Clone the repository**:
    ```bash
    git clone <repository-url>
    cd stalesgd_afl_comparison
    ```
2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: Ensure PyTorch/Torchvision versions match your system, especially CUDA if using GPU).*

## Configuration (`config.yaml`)

Experiments are controlled via a YAML configuration file (e.g., `config.yaml`). Key sections include:

* `experiment_name`: A name for the experiment run.
* `seed`: Base random seed for reproducibility.
* `repetitions`: How many times to repeat the experiment with incrementing seeds.
* `dataset`: Specifies dataset name (`cifar10`), root directory, and partitioning details (`num_clients`, `type: dirichlet`, `alpha`).
* `model`: Specifies model name (`resnet18_cifar`, `cnn_cifar`) and `num_classes`.
* `simulation`: Contains settings for the simulation run:
    * `max_server_iterations` / `max_applied_updates` / `max_wall_time_seconds`: Stopping criteria.
    * `eval_interval`: Frequency of evaluating the global model on the test set.
    * Client settings: `local_epochs`, `batch_size`, `optimizer` (name and params like `lr`).
    * `delay_config`: Defines the delay simulation (`distribution_type` and its `params`). Remember to call `assign_client_speeds` via the manager if using a 'mixed' type.
* `algorithms`: A list of algorithms to run. Each entry specifies:
    * `name`: Matches a key in `ALGORITHM_MAP` in `experiment_manager.py`.
    * `params`: Algorithm-specific hyperparameters (e.g., `client_fraction` for FedAvg, `buffer_size_k` for FedBuff/CA2FL, `server_lr` for async methods, `tau_algo` for AC-AFL-BDA, `malenia_S` for MaleniaSGD).
* `results_dir`: Base directory to save experiment results.

## Running Experiments

To run the experiments defined in a configuration file:

```bash
python main.py --config path/to/your/config.yaml
```

The script will:
1. Create a timestamped subdirectory within the specified `results_dir`.
2. Save the used configuration file (`config.yaml`) into that subdirectory.
3. Run each algorithm specified in the config for the specified number of repetitions.
4. For each run, save the results (logged metrics like accuracy, loss, time, iteration count) as a `.pt` file (PyTorch serialized dictionary) in the run's subdirectory.

## Implemented Algorithms

Located in the `algorithms/` directory:

* **`fedavg.py` (FedAvgServer)**: Standard synchronous Federated Averaging. Selects a fraction of clients per round, waits for all, aggregates.
* **`vanilla_asgd.py` (VanillaASGDServer)**: Basic asynchronous SGD. Applies updates (gradients/deltas) immediately upon arrival.
* **`fedbuff.py` (FedBuffServer)**: Buffered asynchronous aggregation. Server waits for `K` updates in a buffer before applying an aggregated update.
* **`ca2fl.py` (CA2FLServer)**: Cache-Aided ASGD. Extends FedBuff by maintaining client caches ($h_i$) and calibrating updates ($\Delta_i - h_i$) before buffering and aggregation.
* **`delay_adaptive.py` (DelayAdaptiveAFLServer)**: Applies updates immediately but uses an adaptive learning rate scaled based on update staleness ($\tau_t$) relative to a threshold ($\tau_C$). Based on Koloskova et al., 2022.
* **`malenia_sgd.py` (MaleniaSGDServer)**: Collects `S` gradients computed based *only* on the current model version ($w^k$) before updating ($w^{k+1} = w^k - \eta g^k$). Discards stale gradients. *Requires client to return gradients.*
* **`stalesgd.py` (StaleSGDServer)**: Implements AC-AFL-BDA. Maintains a cache of the latest gradient ($U_i$) and its model dispatch iteration ($t_i^{start}$) for each client. Aggregates cached gradients from clients whose staleness ($t - t_i^{start}$) is within $\tau_{algo}$. *Requires client to return gradients.*



