# main.py

import argparse
import os
import sys

# Add project root to Python path to allow importing modules
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

from experiments.experiment_manager import ExperimentManager
# Optional: Import plotting utility if you want to generate plots after run
# from utils.plotting import plot_results 

def main():
    """
    Main function to run Federated Learning experiments.
    Parses command-line arguments for the configuration file,
    initializes the ExperimentManager, and runs the experiments.
    """
    parser = argparse.ArgumentParser(description="Run Federated Learning Experiments")
    parser.add_argument('--config', 
                        type=str, 
                        required=True, 
                        help='Path to the YAML configuration file.')
    # Add other potential arguments here (e.g., --device, --output_dir override)
    
    args = parser.parse_args()

    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file not found at {args.config}")
        sys.exit(1)

    print(f"Loading configuration from: {args.config}")
    
    try:
        # Initialize and run experiments
        manager = ExperimentManager(config_path=args.config)
        all_run_results = manager.run_all_experiments()
        
        print("\nAll experiments completed.")
        
        # --- Optional: Plotting ---
        # If a plotting function exists in utils/plotting.py:
        # print("\nGenerating plots...")
        # try:
        #     # Assume plot_results takes the directory where results were saved
        #     plot_results(manager.experiment_run_dir) 
        #     print(f"Plots saved in {manager.experiment_run_dir}")
        # except Exception as e:
        #      print(f"Error during plotting: {e}")
        # --------------------------
        
    except Exception as e:
        print(f"\nAn error occurred during the experiment:")
        print(e)
        import traceback
        traceback.print_exc() # Print detailed traceback
        sys.exit(1)

if __name__ == "__main__":
    main()