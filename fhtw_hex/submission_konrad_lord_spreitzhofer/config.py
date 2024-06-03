BOARD_SIZE = 4

#MCTS
EPOCHS = 50
NUM_OF_GAMES_PER_EPOCH = 30
MCTS_SIMULATIONS = 600
TEMPERATURE = 1.0

#Neural Network
POLICY_LOSS_WEIGHT = 1.0  # Initial weight for policy loss
VALUE_LOSS_WEIGHT = 1.0  # Initial weight for value loss
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
STEP_SIZE = 5
GAMMA = 0.90
BATCH_SIZE = 32
WARMUP_EPOCHS = 5  # Number of warm-up epochs
WARMUP_LEARNING_RATE = 1e-6  # Initial learning rate for warm-up


#Evaluation

NUM_OF_OPPONENTS_PER_CHECKPOINT = 3
CHECKPOINT_INTERVAL = 20
NUM_OF_GAMES_PER_CHECKPOINT = 15



PARALLEL_GAMES = True # Set to True to parallelize games in training
NUM_PARALLEL_THREADS = 10 # Number of parallel threads to use (adjust based on CPU)
                         # if number is higher than avaiable threads than max threads will be used


# MODEL = 'fhtw_hex/submission_konrad_lord_spreitzhofer/models/final/best_hex_model.keras'  # for the final submission
MODEL = 'fhtw_hex/submission_konrad_lord_spreitzhofer/models/2024-06-03-11-36-13/best_loss/best_hex_model.pth'  # enter the folder to test


