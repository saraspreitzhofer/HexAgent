BOARD_SIZE = 3

#MCTS
EPOCHS = 60
NUM_OF_GAMES_PER_EPOCH = 30
MCTS_SIMULATIONS = 300
TEMPERATURE = 1.0
EPSILON_START = 0.4  # Startvalue for epsilon-greedy strategie
EPSILON_END = 0.1    # Endvalue  for epsilon-greedy strategie


#Neural Network
POLICY_LOSS_WEIGHT = 1.0  # Initial weight for policy loss
VALUE_LOSS_WEIGHT = 1.0  # Initial weight for value loss
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-3
STEP_SIZE = 5
GAMMA = 0.90
BATCH_SIZE = 32
WARMUP_EPOCHS = 10            # Number of warm-up epochs
WARMUP_LEARNING_RATE = 1e-5  # Initial learning rate for warm-up will switch to actual learning_rate
RANDOM_EPOCHS = 7            # How many epochs do we start to train against random agent

#Evaluation

EVALUATION_INTERVAL = 4           # In which epochs do we evaluate
CHECKPOINT_INTERVAL = 8            # When to save an Agent as opponent
NUM_OF_GAMES_PER_CHECKPOINT = 10    # How many games do we play to evaluate
NUM_OF_AGENTS = 4                  # Number of agents to save (excluded Random Agent)



PARALLEL_GAMES = True # Set to True to parallelize games in training
NUM_PARALLEL_THREADS = 24 # Number of parallel threads to use (adjust based on CPU)
                         # if number is higher than avaiable threads than max threads will be used



MODEL = 'fhtw_hex/submission_konrad_lord_spreitzhofer/models/2024-06-08-02-52/best_loss/best_hex_model.pth'  # enter the folder to test


