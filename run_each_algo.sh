#!/bin/bash

# Paths
CONFIG_ORIG="config.yaml"
CONFIG_DIR="generated_configs"
MAIN_SCRIPT="main.py"

# Clear the generated_configs folder before starting
rm -rf "$CONFIG_DIR"
# Create output dir for generated config files
mkdir -p "$CONFIG_DIR"

# Activate virtual environment
source venv/bin/activate

# Python script to split the config into one per algorithm
python3 - <<EOF
import yaml
import os

# Load original config
with open("$CONFIG_ORIG", "r") as f:
    full_config = yaml.safe_load(f)

# Extract and remove the algorithms section
algos = full_config.pop("algorithms", [])

# Save the shared base config
for algo in algos:
    algo_name = algo["name"]
    new_config = dict(full_config)  # shallow copy is fine since we don't mutate
    new_config["algorithms"] = [algo]

    config_path = os.path.join("$CONFIG_DIR", f"config_{algo_name}.yaml")
    with open(config_path, "w") as out:
        yaml.dump(new_config, out, sort_keys=False)
EOF

# Launch each config in parallel
for config_file in "$CONFIG_DIR"/*.yaml; do
    algo_name=$(basename "$config_file" .yaml | sed 's/^config_//')

    # Skip if algo name is "base"
    if [[ "$algo_name" == "base" ]]; then
        echo "Skipping base algorithm..."
        continue
    fi

    # Get current timestamp
    timestamp=$(date +"%Y%m%d_%H%M%S")
  
    log_file="logs/${timestamp}_${algo_name}.log"
    echo "Launching: $algo_name → $log_file"
    mkdir -p logs
    PYTHONUNBUFFERED=1 python "$MAIN_SCRIPT" --config "$config_file" > "$log_file" 2>&1 &
    # # Output the command instead of running it
    # echo "PYTHONUNBUFFERED=1 python $MAIN_SCRIPT --config $config_file > $log_file 2>&1 &"
done


# Wait for all background processes
wait

echo "✅ All algorithms completed."