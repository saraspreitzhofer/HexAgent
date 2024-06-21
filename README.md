# Code Documentation for Hex Agent Training

This document provides an overview of the code structure and functionality to help understand and replicate the process of training a Hex game agent using Monte Carlo Tree Search (MCTS) and a neural network. By following this documentation, you should be able to understand the process of training a Hex game agent using the provided codebase. Adjust configurations as needed to suit your specific requirements and computational resources.

## Prerequisites

Ensure that the following packages are installed:
- `numpy`
- `torch`
- `matplotlib`
- `tqdm`

## Watch or Play Against the Agent

We have provided the functionality to watch the agent play or challenge the agent yourself across different board sizes (3x3, 4x4, 5x5, and 7x7). To do so, simply run `script.py`, select a board size from the terminal prompt, and enjoy the game. You can either watch the agent play against itself, against a random agent, or you can play against the agent yourself. Have fun!

## Configuration

In `config.py`, you can adjust various parameters such as `BOARD_SIZE`, `MCTS_SIMULATIONS`, the number of learning games per epoch, and many other settings. For a quick start, you can use the default parameters, which should work on most computers. Otherwise, note that MCTS with a neural network is computationally intensive.

## Training the Agent

Once the configurations are set, you can start training the agent by running `train.py`. This script coordinates the training process, leveraging various modules and functions to create and optimize the Hex game agent. For more detailed information, please proceed to the next page.

## Module Overview

The project consists of several key modules, each responsible for specific aspects of the training and gameplay process.

### facade.py

This module acts as a facade, providing simplified access to the core functionalities. It includes the following key functions and classes:
- **Node**: Represents a node in the MCTS, storing state, actions, and statistics.
- **ResidualBlock**: A neural network block used to enhance feature extraction.
- **HexNet**: The main neural network architecture for processing the board state and outputting policy and value predictions.
- **create_model**: Creates an instance of HexNet with the specified board size and dropout rate.
- **MCTS**: Implements the Monte Carlo Tree Search algorithm, utilizing the neural network for policy and value estimations.
- **SumTree**: A data structure for prioritized experience replay.
- **PrioritizedReplayBuffer**: Manages the replay buffer with prioritized sampling based on TD error.
- **RandomAgent**: A simple agent that selects random actions.

### train.py

This script orchestrates the training process, calling various functions from the facade and utils modules. Key functions include:
- **play_game**: Simulates a game between the MCTS agent and an opponent (either itself or a random agent).
- **play_games**: Manages multiple game simulations in parallel to generate training data.
- **play_validation**: Evaluates the current agent against previous versions or random agents.
- **validate_against_checkpoints**: Periodically validates the agent against saved checkpoints to ensure continuous improvement.
- **compute_td_error**: Calculates the temporal-difference error for experience prioritization.
- **train_model**: The main training loop that iteratively improves the agent by generating experiences, updating the replay buffer, and training the neural network.

### utils.py

This module provides utility functions for saving and loading models, managing the device setup, and visualizing results. Key functions include:
- **save_results**: Saves training results, including loss and win rate plots.
- **save_config_to_file**: Saves the current configuration settings to a file.
- **save_checkpoint**: Saves a model checkpoint during training.
- **load_checkpoint**: Loads a model checkpoint for further training or evaluation.
- **setup_device**: Sets up the device (CPU or GPU) for training.

##
