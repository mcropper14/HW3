#!/bin/bash

# Run all models with specified configurations
# To run this script in the background and ensure it continues if the connection is interrupted:
# 1. Make the script executable: chmod +x run_all_models.sh
# 2. Run in the background with nohup: nohup ./run_all_models.sh > output.log &

# CNN Model - MNIST


python main.py --model cnn --activation_fn elu
python main.py --model cnn --activation_fn leaky_relu
python main.py --model cnn


# Notify completion

echo "All models have finished training."