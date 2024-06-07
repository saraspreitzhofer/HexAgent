BOARD_SIZE = 3

#MCTS
EPOCHS = 12
NUM_OF_GAMES_PER_EPOCH = 10
MCTS_SIMULATIONS = 20
TEMPERATURE = 1.0
EPSILON_START = 0.3  # Startvalue for epsilon-greedy strategie
EPSILON_END = 0.1    # Endvalue  for epsilon-greedy strategie


#Neural Network
POLICY_LOSS_WEIGHT = 1.3  # Initial weight for policy loss
VALUE_LOSS_WEIGHT = 1.0  # Initial weight for value loss
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
STEP_SIZE = 5
GAMMA = 0.90
BATCH_SIZE = 32
WARMUP_EPOCHS = 3            # Number of warm-up epochs
WARMUP_LEARNING_RATE = 1e-6  # Initial learning rate for warm-up will switch to actual learning_rate
RANDOM_EPOCHS = 2            # How many epochs do we start to train agains random agent

#Evaluation

EVALUATION_INTERVAL = 2            # In which epochs do we evaluate
CHECKPOINT_INTERVAL = 2            # When to save an Agent as opponent
NUM_OF_GAMES_PER_CHECKPOINT = 8    # How many games do we play to evaluate
NUM_OF_AGENTS = 1                  # Number of agents to save (excluded Random Agent)
WINDOW_SIZE = 2                    # Parameter for moving average


PARALLEL_GAMES = True # Set to True to parallelize games in training
NUM_PARALLEL_THREADS = 24 # Number of parallel threads to use (adjust based on CPU)
                         # if number is higher than avaiable threads than max threads will be used


# MODEL = 'fhtw_hex/submission_konrad_lord_spreitzhofer/models/final/best_hex_model.keras'  # for the final submission
MODEL = 'fhtw_hex/submission_konrad_lord_spreitzhofer/models/2024-06-03-11-36-13/best_loss/best_hex_model.pth'  # enter the folder to test


