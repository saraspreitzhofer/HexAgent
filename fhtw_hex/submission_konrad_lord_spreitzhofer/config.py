BOARD_SIZE = 4

#MCTS
EPOCHS = 20
NUM_OF_GAMES_PER_EPOCH = 5
MCTS_SIMULATIONS = 5
TEMPERATURE = 1.0
EPSILON_START = 0.3  # Startvalue for epsilon-greedy strategie
EPSILON_END = 0.05    # Endvalue  for epsilon-greedy strategie


#Neural Network
POLICY_LOSS_WEIGHT = 1.0  # Initial weight for policy loss
VALUE_LOSS_WEIGHT = 1.0  # Initial weight for value loss
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-3
STEP_SIZE = 5
GAMMA = 0.90
BATCH_SIZE = 32
WARMUP_EPOCHS = 10           # Number of warm-up epochs
WARMUP_LEARNING_RATE = 1e-4  # Initial learning rate for warm-up will switch to actual learning_rate
RANDOM_EPOCHS = 20            # How many epochs do we start to train against random agent

#Evaluation

EVALUATION_INTERVAL = 5         # In which epochs do we evaluate
CHECKPOINT_INTERVAL = 20            # When to save an Agent as opponent
NUM_OF_GAMES_PER_CHECKPOINT = 15    # How many games do we play to evaluate
NUM_OF_AGENTS = 3                  # Number of agents to save (excluded Random Agent)


REPLAY_BUFFER_CAPACITY = 100000    # Saving and using moves from past
PARALLEL_GAMES = True # Set to True to parallelize games in training
NUM_PARALLEL_THREADS = 10 # Number of parallel threads to use (adjust based on CPU)
                         # if number is higher than avaiable threads than max threads will be used



MODEL = 'fhtw_hex/submission_konrad_lord_spreitzhofer/models/3x3_v1/best_loss/best_hex_model.pth'  # enter the folder to test


