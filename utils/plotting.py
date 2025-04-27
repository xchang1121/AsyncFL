# utils/plotting.py

import torch
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

# Optional: Customize plot style
# import seaborn as sns
# sns.set_theme(style="whitegrid")

def load_results(results_dir: str) -> Dict[str, pd.DataFrame]:
    """
    Loads all experiment result files (.pt) from a directory.

    Args:
        results_dir (str): The directory containing the saved .pt result files.

    Returns:
        Dict[str, pd.DataFrame]: A dictionary where keys are algorithm names 
                                 and values are pandas DataFrames containing the 
                                 results for that algorithm across repetitions.
    """
    all_results = {}
    result_files = glob.glob(os.path.join(results_dir, "*_results.pt"))

    if not result_files:
        print(f"Warning: No result files found in {results_dir}")
        return {}

    print(f"Found result files: {[os.path.basename(f) for f in result_files]}")

    for f_path in result_files:
        try:
            results_list = torch.load(f_path) # Assumes saved object is List[Dict[str, List]]
            if not isinstance(results_list, list):
                 print(f"Warning: Expected a list of results in {f_path}, found {type(results_list)}. Skipping.")
                 continue
                 
            base_filename = os.path.basename(f_path)
            # Extract algorithm name (assuming format like xxx.pt)
            algo_name = base_filename.split('_seed')[0] 
            
            print(f"Loading results for {algo_name} from {base_filename}...")

            # Combine results from multiple repetitions into a single DataFrame
            df_list = []
            for rep_idx, single_run_results in enumerate(results_list):
                 # Find the primary key for x-axis (iteration or applied_updates)
                 if 'server_iteration' in single_run_results and single_run_results['server_iteration']:
                      iter_key = 'server_iteration'
                 elif 'applied_updates' in single_run_results and single_run_results['applied_updates']:
                      iter_key = 'applied_updates'
                 else:
                     print(f"Warning: No iteration or applied_updates key found in results for {algo_name}, rep {rep_idx}. Skipping rep.")
                     continue
                     
                 # Check for length consistency
                 lengths = {k: len(v) for k, v in single_run_results.items()}
                 if len(set(lengths.values())) > 1:
                      print(f"Warning: Inconsistent lengths in results for {algo_name}, rep {rep_idx}. Skipping rep. Lengths: {lengths}")
                      # Attempt to truncate to minimum length? Or skip. Skipping is safer.
                      continue

                 df = pd.DataFrame(single_run_results)
                 df['repetition'] = rep_idx
                 df['algorithm'] = algo_name
                 df_list.append(df)

            if df_list:
                if algo_name not in all_results:
                     all_results[algo_name] = pd.concat(df_list, ignore_index=True)
                else:
                     # Append if algo exists from multiple files (e.g., if saving logic changes)
                     all_results[algo_name] = pd.concat([all_results[algo_name], pd.concat(df_list, ignore_index=True)], ignore_index=True)
            
        except Exception as e:
            print(f"Error loading or processing file {f_path}: {e}")

    return all_results


def plot_metric_vs_time(results_dfs: Dict[str, pd.DataFrame], 
                        metric: str, 
                        time_axis: str, 
                        title: str, 
                        output_filename: str,
                        smoothing_window: Optional[int] = None):
    """
    Generates a plot comparing a metric vs. time across algorithms.

    Args:
        results_dfs: Dictionary of DataFrames, key=algo_name, value=DataFrame.
        metric: The column name of the metric to plot (e.g., 'test_accuracy', 'test_loss').
        time_axis: The column name for the x-axis ('server_iteration', 'applied_updates', 'wall_clock_time').
        title: The title for the plot.
        output_filename: Path to save the plot image.
        smoothing_window: Optional window size for simple moving average smoothing.
    """
    plt.figure(figsize=(10, 6))

    for algo_name, df in results_dfs.items():
        if time_axis not in df.columns or metric not in df.columns:
            print(f"Warning: Skipping {algo_name} for plot '{title}'. Missing columns: {time_axis} or {metric}")
            continue
            
        # Aggregate results across repetitions (mean +/- std deviation)
        # Group by the time axis and calculate mean/std for the metric
        agg_data = df.groupby(time_axis)[metric].agg(['mean', 'std']).reset_index()
        
        line, = plt.plot(agg_data[time_axis], agg_data['mean'], label=algo_name)
        plt.fill_between(agg_data[time_axis], 
                         agg_data['mean'] - agg_data['std'], 
                         agg_data['mean'] + agg_data['std'], 
                         alpha=0.2, color=line.get_color())

        # Optional Smoothing
        if smoothing_window and len(agg_data['mean']) >= smoothing_window:
             smoothed_mean = agg_data['mean'].rolling(window=smoothing_window, min_periods=1, center=True).mean()
             plt.plot(agg_data[time_axis], smoothed_mean, linestyle='--', color=line.get_color(), label=f'{algo_name} (smoothed)')


    plt.xlabel(time_axis.replace('_', ' ').title())
    plt.ylabel(metric.replace('_', ' ').title())
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    print(f"Plot saved to {output_filename}")


def plot_all_results(results_dir: str):
    """
    Loads results from a directory and generates standard comparison plots.
    """
    all_results_dfs = load_results(results_dir)
    if not all_results_dfs:
        print("No results loaded, skipping plotting.")
        return

    plot_dir = os.path.join(results_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    # Determine primary iteration key (preference: server_iteration > applied_updates)
    # Assumes at least one df exists
    sample_df = next(iter(all_results_dfs.values()))
    iteration_key = None
    if 'server_iteration' in sample_df.columns:
         iteration_key = 'server_iteration'
    elif 'applied_updates' in sample_df.columns:
         iteration_key = 'applied_updates'
         
    if iteration_key:
        # Plot Accuracy vs Iteration
        plot_metric_vs_time(all_results_dfs, 
                            metric='test_accuracy', 
                            time_axis=iteration_key, 
                            title=f'Test Accuracy vs {iteration_key.replace("_", " ").title()}',
                            output_filename=os.path.join(plot_dir, f'accuracy_vs_{iteration_key}.png'))
        
        # Plot Loss vs Iteration
        plot_metric_vs_time(all_results_dfs, 
                            metric='test_loss', 
                            time_axis=iteration_key, 
                            title=f'Test Loss vs {iteration_key.replace("_", " ").title()}',
                            output_filename=os.path.join(plot_dir, f'loss_vs_{iteration_key}.png'))
    else:
         print("Warning: Cannot plot vs iterations - key not found.")


    # Plot Accuracy vs Wall Clock Time
    plot_metric_vs_time(all_results_dfs, 
                        metric='test_accuracy', 
                        time_axis='wall_clock_time', 
                        title='Test Accuracy vs Wall Clock Time',
                        output_filename=os.path.join(plot_dir, 'accuracy_vs_wall_time.png'))
    
    # Plot Loss vs Wall Clock Time
    plot_metric_vs_time(all_results_dfs, 
                        metric='test_loss', 
                        time_axis='wall_clock_time', 
                        title='Test Loss vs Wall Clock Time',
                        output_filename=os.path.join(plot_dir, 'loss_vs_wall_time.png'))
    
    # Plot Average Training Loss (if available)
    if 'train_loss' in sample_df.columns:
         if iteration_key:
              plot_metric_vs_time(all_results_dfs, 
                                  metric='train_loss', 
                                  time_axis=iteration_key, 
                                  title=f'Avg Train Loss vs {iteration_key.replace("_", " ").title()}',
                                  output_filename=os.path.join(plot_dir, f'train_loss_vs_{iteration_key}.png'))
         plot_metric_vs_time(all_results_dfs, 
                             metric='train_loss', 
                             time_axis='wall_clock_time', 
                             title='Avg Train Loss vs Wall Clock Time',
                             output_filename=os.path.join(plot_dir, 'train_loss_vs_wall_time.png'))


# Example Usage (called from main.py after experiments)
# if __name__ == '__main__':
#     # Assumes results are saved in './results/experiment_name_timestamp/'
#     example_results_dir = './results/cifar10_noniid_comparison_20250425_120000' # Replace with actual dir
#     if os.path.exists(example_results_dir):
#          plot_all_results(example_results_dir)
#     else:
#          print(f"Example results directory not found: {example_results_dir}")