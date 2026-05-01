#!/bin/bash

# 1. Deactivate any lingering virtual environments from other folders
if [ -n "$VIRTUAL_ENV" ]; then
    deactivate
fi

# 2. Purge and load the correct HPC modules
module purge
module load python3/3.11.10
module load cuda/12.4
module load cudnn/v8.9.7.29-prod-cuda-12.X

# 3. Route the uv cache to your high-capacity work3 drive!
export UV_CACHE_DIR="/work3/s215141/.uv_cache"

# 4. Activate this project's specific virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
    echo "Mini-Project-3 Environment Loaded! Ready to work!"
else
    echo "Warning: .venv not found. Run 'uv sync' to build the environment."
fi