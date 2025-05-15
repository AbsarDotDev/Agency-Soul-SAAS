#!/bin/bash

# Exit on error
set -e

echo "Activating conda environment 'nomessos'..."
eval "$(conda shell.bash hook)"
conda activate nomessos

echo "Starting FastAPI application..."
python main.py 