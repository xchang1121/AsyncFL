# utils/simulation.py

import numpy as np
import random
from typing import List, Dict, Any, Literal

# Global variable to store client speed assignments for mixed mode
# This ensures consistency across calls for the same client set
_client_speed_groups: Dict[int, Literal['fast', 'slow']] = {}

def assign_client_speeds(client_ids: List[int], slow_fraction: float = 0.2):
    """
    Assigns clients to 'fast' or 'slow' groups for mixed mode simulation.

    Args:
        client_ids: List of all client IDs.
        slow_fraction: Fraction of clients designated as 'slow'.
    """
    global _client_speed_groups
    _client_speed_groups = {} # Reset for potentially new client set
    n_clients = len(client_ids)
    n_slow = int(n_clients * slow_fraction)
    
    shuffled_ids = random.sample(client_ids, n_clients)
    
    for i, client_id in enumerate(shuffled_ids):
        if i < n_slow:
            _client_speed_groups[client_id] = 'slow'
        else:
            _client_speed_groups[client_id] = 'fast'
    print(f"Assigned {n_slow} clients as slow and {n_clients - n_slow} as fast.")


def simulate_delay(client_id: int,
                   distribution_type: Literal['exponential', 'mixed_uniform', 'mixed_exponential'] = 'exponential',
                   params: Dict[str, Any] = None) -> float:
    """
    Simulates the computation and communication delay for a client update.

    Args:
        client_id: The ID of the client.
        distribution_type: The type of delay distribution to use.
            - 'exponential': Exponential distribution with mean beta. Requires 'beta'.
            - 'mixed_uniform': Mixture of uniform distributions for fast/slow clients.
                               Requires 'fast_min', 'fast_max', 'slow_min', 'slow_max'.
            - 'mixed_exponential': Mixture of exponential distributions for fast/slow clients.
                                   Requires 'fast_beta', 'slow_beta'.
        params: Dictionary containing parameters for the chosen distribution.

    Returns:
        The simulated delay time (non-negative).
    """
    if params is None:
        params = {}

    delay = 0.0
    
    # Basic fixed delay component (t_fix_i in the request) - set to 0 for now
    t_fix = params.get('t_fix', 0.0) 

    if distribution_type == 'exponential':
        beta = params.get('beta', 1.0)  # Default average delay
        if beta <= 0:
            raise ValueError("Parameter 'beta' for exponential distribution must be positive.")
        # Ensure delay is non-negative
        delay = t_fix + max(0, np.random.exponential(scale=beta))
        
    elif distribution_type == 'mixed_uniform':
        if not _client_speed_groups:
             raise RuntimeError("Client speed groups not assigned. Call assign_client_speeds first.")
        
        client_speed = _client_speed_groups.get(client_id)
        if client_speed is None:
             raise ValueError(f"Client ID {client_id} not found in assigned speed groups.")

        fast_min = params.get('fast_min', 0.1)
        fast_max = params.get('fast_max', 1.0)
        slow_min = params.get('slow_min', 2.0)
        slow_max = params.get('slow_max', 5.0)

        if not (0 <= fast_min <= fast_max and 0 <= slow_min <= slow_max):
             raise ValueError("Uniform distribution parameters must be non-negative and min <= max.")

        if client_speed == 'fast':
            delay = t_fix + np.random.uniform(fast_min, fast_max)
        else: # slow
            delay = t_fix + np.random.uniform(slow_min, slow_max)
            
    elif distribution_type == 'mixed_exponential':
        if not _client_speed_groups:
             raise RuntimeError("Client speed groups not assigned. Call assign_client_speeds first.")

        client_speed = _client_speed_groups.get(client_id)
        if client_speed is None:
             raise ValueError(f"Client ID {client_id} not found in assigned speed groups.")

        fast_beta = params.get('fast_beta', 0.5)
        slow_beta = params.get('slow_beta', 5.0)
        
        if fast_beta <= 0 or slow_beta <= 0:
            raise ValueError("Parameters 'fast_beta' and 'slow_beta' for mixed exponential distribution must be positive.")

        if client_speed == 'fast':
            delay = t_fix + max(0, np.random.exponential(scale=fast_beta))
        else: # slow
            delay = t_fix + max(0, np.random.exponential(scale=slow_beta))
            
    else:
        raise ValueError(f"Unknown delay distribution type: {distribution_type}")

    return delay

# Example Usage (can be removed or placed in a test file)
if __name__ == '__main__':
    num_clients = 100
    client_ids = list(range(num_clients))
    
    # --- Exponential Example ---
    print("--- Exponential Delay Simulation ---")
    exp_params = {'beta': 2.5, 't_fix': 0.1}
    delays_exp = [simulate_delay(cid, 'exponential', exp_params) for cid in client_ids]
    print(f"Sample Exponential Delays (beta=2.5, t_fix=0.1): {delays_exp[:10]}")
    print(f"Average delay: {np.mean(delays_exp):.2f}\n")

    # --- Mixed Uniform Example ---
    print("--- Mixed Uniform Delay Simulation ---")
    assign_client_speeds(client_ids, slow_fraction=0.3) # Assign 30% as slow
    mix_uni_params = {'fast_min': 0.5, 'fast_max': 1.5, 'slow_min': 4.0, 'slow_max': 8.0}
    delays_mix_uni = [simulate_delay(cid, 'mixed_uniform', mix_uni_params) for cid in client_ids]
    print(f"Sample Mixed Uniform Delays: {delays_mix_uni[:10]}")
    # Separate delays by group to check average
    fast_delays_uni = [d for i, d in enumerate(delays_mix_uni) if _client_speed_groups[client_ids[i]] == 'fast']
    slow_delays_uni = [d for i, d in enumerate(delays_mix_uni) if _client_speed_groups[client_ids[i]] == 'slow']
    if fast_delays_uni: print(f"Average fast delay: {np.mean(fast_delays_uni):.2f}")
    if slow_delays_uni: print(f"Average slow delay: {np.mean(slow_delays_uni):.2f}\n")

    # --- Mixed Exponential Example ---
    print("--- Mixed Exponential Delay Simulation ---")
    # Use the same speed assignment as above
    mix_exp_params = {'fast_beta': 0.8, 'slow_beta': 6.0}
    delays_mix_exp = [simulate_delay(cid, 'mixed_exponential', mix_exp_params) for cid in client_ids]
    print(f"Sample Mixed Exponential Delays: {delays_mix_exp[:10]}")
    fast_delays_exp = [d for i, d in enumerate(delays_mix_exp) if _client_speed_groups[client_ids[i]] == 'fast']
    slow_delays_exp = [d for i, d in enumerate(delays_mix_exp) if _client_speed_groups[client_ids[i]] == 'slow']
    if fast_delays_exp: print(f"Average fast delay: {np.mean(fast_delays_exp):.2f}")
    if slow_delays_exp: print(f"Average slow delay: {np.mean(slow_delays_exp):.2f}\n")