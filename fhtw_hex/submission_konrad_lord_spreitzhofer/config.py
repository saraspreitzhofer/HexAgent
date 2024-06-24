BOARD_SIZE = 3            # This needs to be adjusted for train.py and script.py (which agent one want to train or play)

#MCTS
EPOCHS = 30
NUM_OF_GAMES_PER_EPOCH = 25
MCTS_SIMULATIONS = 25
TEMPERATURE_START = 1.0
TEMPERATURE_END = 0.1
EPSILON_START = 0.6      # Startvalue for epsilon-greedy strategie
EPSILON_END = 0.08        # Endvalue  for epsilon-greedy strategie


#Neural Network
POLICY_LOSS_WEIGHT = 1.00    # Initial weight for policy loss
VALUE_LOSS_WEIGHT = 1.0      # Initial weight for value loss
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.01
BATCH_SIZE = 64
WARMUP_EPOCHS = 10              # Number of warm-up epochs
WARMUP_LEARNING_RATE = 0.0001   # Initial learning rate for warm-up will switch to actual learning_rate
RANDOM_EPOCHS = 20              # How many epochs do we start to train against random agent

#Evaluation

EVALUATION_INTERVAL = 5            # In which epochs do we evaluate
CHECKPOINT_INTERVAL = 10            # When to save an Agent as opponent
NUM_OF_GAMES_PER_CHECKPOINT = 20   # How many games do we play to evaluate
NUM_OF_AGENTS = 1                  # Number of agents to save (excluded Random Agent)


REPLAY_BUFFER_CAPACITY = 60000     # Saving and using moves from past
PARALLEL_GAMES = True              # Set to True to parallelize games in training
NUM_PARALLEL_THREADS = 10          # Number of parallel threads to use (adjust based on CPU)
                                   # if number is higher than avaiable threads than max threads will be used





