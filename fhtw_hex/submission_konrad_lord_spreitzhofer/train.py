import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from datetime import datetime
from torch.optim.lr_scheduler import StepLR
from multiprocessing import Pool
import torch.multiprocessing as mp
from random import choice
from collections import deque
import random
from fhtw_hex import hex_engine as engine
from fhtw_hex.submission_konrad_lord_spreitzhofer import config as base_config
from fhtw_hex.submission_konrad_lord_spreitzhofer.utils import load_checkpoint, save_checkpoint, save_config_to_file, save_results, setup_device

# Convert config module to a dictionary(this should enable to train on cluster parallel)
base_config_dict = {key: getattr(base_config, key) for key in dir(base_config) if not key.startswith("__")}
config = deepcopy(base_config_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Node:
    hex_position_class = engine.HexPosition

    def __init__(self, state, action_space=None, parent=None, action=None, prior=0):
        self.state = state
        self.action_space = action_space
        self.parent = parent
        self.action = action
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.hex_position = self.create_hex_position(state)

    def create_hex_position(self, state):
        temp_hex_position = self.hex_position_class(size=len(state))
        temp_hex_position.board = state
        temp_hex_position.player = 1 if sum(sum(row) for row in state) == 0 else -1  # Determine the player
        return temp_hex_position

    def is_terminal(self):
        return self.hex_position.winner != 0

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, policy):
        valid_actions = self.hex_position.get_action_space()  # Ensure only valid moves are considered
        for action, prob in zip(self.action_space, policy):
            if action in valid_actions:  # Check if the action is valid
                new_state = deepcopy(self.state)
                temp_hex_position = self.create_hex_position(new_state)
                temp_hex_position.moove(action)
                self.children.append(
                    Node(temp_hex_position.board, parent=self, action=action, prior=prob,
                         action_space=self.action_space)
                )

    def select_child(self, exploration_weight=1.0):
        return max(self.children, key=lambda x: x.value(exploration_weight))

    def update(self, value):
        self.visit_count += 1
        self.value_sum += value

    def value(self, exploration_weight=1.0):
        epsilon = 1e-6
        if self.visit_count == 0:
            return float('inf')
        exploitation = self.value_sum / (self.visit_count + epsilon)
        exploration = exploration_weight * self.prior * np.sqrt(
            np.log(self.parent.visit_count + 1) / (self.visit_count + epsilon))
        return exploitation + exploration

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class HexNet(nn.Module):
    def __init__(self, board_size):
        super(HexNet, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.residual_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(3)]) #Anzahl der Blöcke
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.1)  # Dropout-Schicht mit einer Dropout-Rate von 30%
        self.policy_head = nn.Linear(64 * board_size * board_size, board_size * board_size)
        self.value_head = nn.Linear(64 * board_size * board_size, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        for block in self.residual_blocks:
            x = block(x)
        x = self.flatten(x)
        x = self.dropout(x)  # Dropout-Schicht anwenden
        policy = F.log_softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value

def create_model(board_size):
    model = HexNet(board_size).to(device)
    return model

class MCTS:
    def __init__(self, model, simulations=100, device=device, epsilon=0.1, board_size=5):
        self.model = model
        self.simulations = simulations
        self.device = device
        self.epsilon = epsilon
        self.board_size = board_size
        self.model.to(self.device)

    def get_action(self, state, action_set):
        root = Node(state, action_set)
        for _ in range(self.simulations):
            self.search(root)
        if np.random.rand() < self.epsilon:
            return np.random.choice(root.children).action
        return max(root.children, key=lambda c: c.visit_count).action

    def search(self, node, exploration_weight=1.0):
        if node.is_terminal():
            return -node.state.winner
        if not node.is_expanded():
            policy, value = self.evaluate(node)
            node.expand(policy)
            return -value
        child = node.select_child(exploration_weight)
        value = self.search(child, exploration_weight)
        node.update(-value)
        return -value

    def evaluate(self, node):
        board = np.array(node.state).reshape((1, 1, self.board_size, self.board_size)).astype(np.float32)
        board = torch.tensor(board, device=self.device)
        self.model.eval()
        with torch.no_grad():
            policy, value = self.model(board)
        policy = np.exp(policy.cpu().numpy()[0] / config['TEMPERATURE'])
        policy = policy / np.sum(policy)
        return policy, value.cpu().numpy()[0][0]

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Log-Funktion
log_buffer = []

def log_message(message):
    print(message)
    log_buffer.append(message)

def save_log_to_file(log_path):
    with open(log_path, 'w') as f:
        for message in log_buffer:
            f.write(message + '\n')

def play_game(mcts: MCTS, board_size: int, opponent='random'):
    game = engine.HexPosition(board_size)
    state_history = []
    while game.winner == 0:
        state_history.append(deepcopy(game.board))
        if game.player == 1:
            chosen = mcts.get_action(game.board, game.get_action_space())
        else:
            chosen = choice(game.get_action_space()) if opponent == 'random' else mcts.get_action(game.board,
                                                                                                  game.get_action_space())
        game.moove(chosen)
    return state_history, game.winner

def play_game_worker(args):
    model_state_dict, board_size, opponent, device, epsilon = args
    model = create_model(board_size).to(device)
    model.load_state_dict(model_state_dict)
    mcts = MCTS(model, epsilon=epsilon, board_size=board_size)
    return play_game(mcts, board_size, opponent)

def play_games(model, board_size, num_games, opponent='random', epsilon=0.1):
    model_state_dict = model.state_dict()
    total_cpus = os.cpu_count()
    if config['PARALLEL_GAMES']:
        num_threads = min(total_cpus, config['NUM_PARALLEL_THREADS'])
        log_message(f"Using {num_threads} parallel threads out of {total_cpus} available")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mp.set_start_method('spawn', force=True)
        with Pool(num_threads) as pool:
            args = [(model_state_dict, board_size, opponent, device, epsilon) for _ in range(num_games)]
            results = list(tqdm(pool.imap(play_game_worker, args), total=num_games, unit='game'))
        return results
    else:
        log_message(f"Parallel games disabled. Using a single thread out of {total_cpus} available")
        results = []
        mcts = MCTS(model, epsilon=epsilon, board_size=board_size)
        for _ in tqdm(range(num_games), unit='game'):
            results.append(play_game(mcts, board_size, opponent))
        return results

class RandomAgent:
    def get_action(self, board, action_space):
        return choice(action_space)

def play_validation(args):
    board_size, current_mcts, checkpoint_mcts, random_agent = args
    game = engine.HexPosition(board_size)
    starter = choice(["current", "checkpoint"])  # Zufällige Auswahl des Startspielers

    if random_agent:
        checkpoint_mcts = RandomAgent()
    player1, player2 = (current_mcts, checkpoint_mcts) if starter == "current" else (checkpoint_mcts, current_mcts)

    first_move_done_by_player2 = False  # Flag, um zu verfolgen, ob player2 seinen ersten Zug gemacht hat

    while game.winner == 0:
        if game.player == 1:  # Spieler 1
            if player1 == checkpoint_mcts and not first_move_done_by_player2:  # Zufälliger erster Zug für Checkpoint-Agent
                chosen = choice(game.get_action_space())
                first_move_done_by_player2 = True
            else:
                chosen = player1.get_action(game.board, game.get_action_space())
        else:  # Spieler 2
            if player2 == checkpoint_mcts and not first_move_done_by_player2:  # Zufälliger erster Zug für Checkpoint-Agent
                chosen = choice(game.get_action_space())
                first_move_done_by_player2 = True
            else:
                chosen = player2.get_action(game.board, game.get_action_space())
        game.moove(chosen)

    move_count = len(game.history)
    result = 1 if (
            (game.winner == 1 and starter == "current") or (game.winner == -1 and starter == "checkpoint")) else 0
    return result, move_count

def validate_against_checkpoints(model, board_size, num_games=10, model_folder='models', checkpoints=[]):
    model.eval()
    current_mcts = MCTS(model, board_size=board_size)
    win_rates = []
    move_rates = []

    # Begrenze die Checkpoints-Liste auf die gewünschte Anzahl von Agenten plus den Random-Agenten
    checkpoints = checkpoints[:config['NUM_OF_AGENTS'] + 1]

    with torch.no_grad():
        for i, checkpoint in enumerate(tqdm(checkpoints, desc='Checkpoints', unit='checkpoint')):  # Including RandomAgent
            if 'random_agent_checkpoint.pth.tar' in checkpoint:
                checkpoint_mcts = RandomAgent()
            else:
                checkpoint_model, _ = load_checkpoint(checkpoint, board_size)
                checkpoint_mcts = MCTS(checkpoint_model, board_size=board_size)

            wins = 0
            total_moves = 0
            random_agent = True if i == 0 else False

            if config['PARALLEL_GAMES']:
                total_cpus = os.cpu_count()
                num_threads = min(total_cpus, config['NUM_PARALLEL_THREADS'])
                mp.set_start_method('spawn', force=True)
                with Pool(num_threads) as pool:
                    args = [(board_size, current_mcts, checkpoint_mcts, random_agent) for _ in range(num_games)]
                    results = list(tqdm(pool.imap(play_validation, args), total=num_games, unit='game'))
                    win_results, move_counts = zip(*results)
                    wins = sum(win_results)
                    total_moves = sum(move_counts)
            else:
                for _ in range(num_games):
                    args = (board_size, current_mcts, checkpoint_mcts, random_agent)
                    win_result, move_count = play_validation(args)
                    wins += win_result
                    total_moves += move_count
            win_rates.append(wins / num_games)
            move_rates.append(total_moves / num_games)

    return win_rates, move_rates

def train_model():
    local_config = deepcopy(config)  # Create a local copy of the configuration

    device = setup_device()
    log_message("Creating model...")
    model = create_model(local_config['BOARD_SIZE']).to(device)
    mcts = MCTS(model, simulations=local_config['MCTS_SIMULATIONS'], epsilon=local_config['EPSILON_START'], board_size=local_config['BOARD_SIZE'])
    best_loss = float('inf')
    best_model_state = None

    models_folder = 'models'
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_folder = os.path.join(models_folder, current_time)
    os.makedirs(model_folder, exist_ok=True)

    # Save a RandomAgent checkpoint
    random_agent_checkpoint_path = os.path.join(model_folder, 'random_agent_checkpoint.pth.tar')
    torch.save({'state_dict': None, 'optimizer': None}, random_agent_checkpoint_path)

    log_message("Saving config to file...")
    save_config_to_file(local_config, filename=os.path.join(model_folder, 'config.py'))

    losses = []
    policy_losses = []
    value_losses = []
    win_rates = [[] for _ in range(local_config['NUM_OF_AGENTS'] + 1)]  # Including Random Agent
    avg_moves = [[] for _ in range(local_config['NUM_OF_AGENTS'] + 1)]  # Including Random Agent

    criterion_policy = nn.KLDivLoss(reduction='batchmean')
    criterion_value = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=local_config['WARMUP_LEARNING_RATE'], weight_decay=local_config['WEIGHT_DECAY'])
    scheduler = StepLR(optimizer, local_config['STEP_SIZE'], gamma=local_config['GAMMA'])
    model.to(device)

    replay_buffer = ReplayBuffer(capacity=10000)

    for epoch in range(1, local_config['EPOCHS'] + 1):
        log_message(f"Starting Epoch {epoch}/{local_config['EPOCHS']}")
        if epoch <= local_config['WARMUP_EPOCHS']:
            lr = local_config['WARMUP_LEARNING_RATE'] + (local_config['LEARNING_RATE'] - local_config['WARMUP_LEARNING_RATE']) * (
                    epoch / local_config['WARMUP_EPOCHS'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif epoch == local_config['WARMUP_EPOCHS'] + 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = local_config['LEARNING_RATE']

        epsilon = local_config['EPSILON_END'] + (local_config['EPSILON_START'] - local_config['EPSILON_END']) * (1 - epoch / local_config['EPOCHS'])
        results = play_games(model, local_config['BOARD_SIZE'], local_config['NUM_OF_GAMES_PER_EPOCH'],
                             opponent='self' if epoch > local_config['RANDOM_EPOCHS'] else 'random', epsilon=epsilon)

        for state_history, result in results:
            for state in state_history:
                policies = np.random.dirichlet(np.ones(local_config['BOARD_SIZE'] * local_config['BOARD_SIZE']))
                replay_buffer.add((state, policies, result))

        if len(replay_buffer) < local_config['BATCH_SIZE']:
            log_message("Not enough samples in replay buffer. Skipping training.")
            continue

        states, policies, values = zip(*replay_buffer.sample(local_config['BATCH_SIZE']))

        states = np.array(states).reshape((-1, 1, local_config['BOARD_SIZE'], local_config['BOARD_SIZE']))
        policies = np.array(policies)
        values = np.array(values).astype(np.float32)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        policies = torch.tensor(policies, dtype=torch.float32).to(device)
        values = torch.tensor(values, dtype=torch.float32).to(device)

        model.train()
        optimizer.zero_grad()

        policy_outputs, value_outputs = model(states)
        policy_loss = criterion_policy(policy_outputs, policies)
        value_loss = criterion_value(value_outputs.squeeze(), values)
        loss = local_config['POLICY_LOSS_WEIGHT'] * policy_loss + local_config['VALUE_LOSS_WEIGHT'] * value_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())

        if epoch == 1 or epoch % local_config['CHECKPOINT_INTERVAL'] == 0:
            checkpoint_epoch = epoch
            save_checkpoint(model, optimizer, checkpoint_epoch, model_folder,
                            filename=f'checkpoint_epoch_{checkpoint_epoch}.pth.tar')

        win_rates_checkpoint, avg_moves_checkpoint = [], []
        if epoch % local_config['EVALUATION_INTERVAL'] == 0 or epoch == local_config['EPOCHS']:
            checkpoints = [random_agent_checkpoint_path] + [os.path.join(model_folder, f'checkpoint_epoch_{e}.pth.tar')
                                                            for e in range(local_config['CHECKPOINT_INTERVAL'], epoch + 1,
                                                                           local_config['CHECKPOINT_INTERVAL'])]
            checkpoints = checkpoints[:local_config['NUM_OF_AGENTS'] + 1]  # Sicherstellen, dass die Liste korrekt begrenzt ist
            log_message(f"Evaluating against checkpoints: {checkpoints}")
            win_rates_checkpoint, avg_moves_checkpoint = validate_against_checkpoints(model, local_config['BOARD_SIZE'],
                                                                                      num_games=local_config['NUM_OF_GAMES_PER_CHECKPOINT'],
                                                                                      model_folder=model_folder,
                                                                                      checkpoints=checkpoints)
            for i, (wr, am) in enumerate(zip(win_rates_checkpoint, avg_moves_checkpoint)):
                win_rates[i].append(wr)
                avg_moves[i].append(am)

        total_loss = policy_loss.item() + value_loss.item()
        log_message(
            f"Completed Epoch {epoch}/{local_config['EPOCHS']} with Loss: {total_loss}, Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")
        log_message(
            f"Random_Agent: Win Rates: {win_rates[0][-1] if win_rates[0] else 'N/A'}, Avg. Moves: {avg_moves[0][-1] if avg_moves[0] else 'N/A'}")
        for i in range(1, len(win_rates)):
            log_message(
                f"Agent_Checkpoint_Epoch_{i * local_config['CHECKPOINT_INTERVAL']}: Win Rates: {win_rates[i][-1] if win_rates[i] else 'N/A'}, Avg. Moves: {avg_moves[i][-1] if avg_moves[i] else 'N/A'}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()

    best_model_path = os.path.join(model_folder, 'best_loss')
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    torch.save(best_model_state, os.path.join(best_model_path, 'best_hex_model.pth'))
    save_results(losses, win_rates, policy_losses, value_losses, best_model_path, avg_moves, checkpoints)

    # Save log to file
    log_path = os.path.join(model_folder, 'train.log')
    save_log_to_file(log_path)
    log_message(f"Logfile created at {log_path}")

def save_results(losses, win_rates, policy_losses, value_losses, best_model_path, avg_moves, checkpoints):
    epochs = len(losses)

    # Plotting loss
    plt.figure()
    plt.plot(range(epochs), losses, label="Total Loss")
    plt.plot(range(epochs), policy_losses, label="Policy Loss")
    plt.plot(range(epochs), value_losses, label="Value Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')
    plt.savefig(os.path.join(best_model_path, 'loss.png'))
    plt.close()

    # Plotting win rates and moves
    for i in range(len(win_rates)):
        agent_win_rates = win_rates[i]
        agent_avg_moves = avg_moves[i]

        # Calculate epochs for this agent
        if i == 0:
            agent_epochs = list(range(config['EVALUATION_INTERVAL'], epochs + 1, config['EVALUATION_INTERVAL']))
            legend_name = 'Random_Agent'
        else:
            start_epoch = config['CHECKPOINT_INTERVAL'] * i
            agent_epochs = list(range(start_epoch, epochs + 1, config['EVALUATION_INTERVAL']))
            legend_name = f'checkpoint_epoch_{start_epoch}'

        plt.figure()
        plt.plot(agent_epochs, agent_win_rates, label=f'{legend_name} Win Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.title('Win Rate over Checkpoints')
        plt.savefig(os.path.join(best_model_path, f'win_rate_{i}.png'))
        plt.close()

        plt.figure()
        plt.plot(agent_epochs, agent_avg_moves, label=f'{legend_name} Avg. Moves')
        plt.xlabel('Epoch')
        plt.ylabel('Moves')
        plt.legend()
        plt.title('Moves over Checkpoints')
        plt.savefig(os.path.join(best_model_path, f'avg_moves_{i}.png'))
        plt.close()

    # Plotting combined win rates and moves
    plt.figure()
    for i in range(len(win_rates)):
        agent_win_rates = win_rates[i]
        if i == 0:
            agent_epochs = list(range(config['EVALUATION_INTERVAL'], epochs + 1, config['EVALUATION_INTERVAL']))
        else:
            start_epoch = config['CHECKPOINT_INTERVAL'] * i
            agent_epochs = list(range(start_epoch, epochs + 1, config['EVALUATION_INTERVAL']))
        label = f'Random_Agent' if i == 0 else f'checkpoint_epoch_{start_epoch}'
        plt.plot(agent_epochs, agent_win_rates, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.title('Win Rate over Checkpoints')
    plt.savefig(os.path.join(best_model_path, 'win_rate_combined.png'))
    plt.close()

    plt.figure()
    for i in range(len(avg_moves)):
        agent_avg_moves = avg_moves[i]
        if i == 0:
            agent_epochs = list(range(config['EVALUATION_INTERVAL'], epochs + 1, config['EVALUATION_INTERVAL']))
        else:
            start_epoch = config['CHECKPOINT_INTERVAL'] * i
            agent_epochs = list(range(start_epoch, epochs + 1, config['EVALUATION_INTERVAL']))
        label = f'Random_Agent' if i == 0 else f'checkpoint_epoch_{start_epoch}'
        plt.plot(agent_epochs, agent_avg_moves, label=label)
    plt.xlabel('Epoch')
    plt.ylabel('Moves')
    plt.legend()
    plt.title('Moves over Checkpoints')
    plt.savefig(os.path.join(best_model_path, 'avg_moves_combined.png'))
    plt.close()

if __name__ == "__main__":
    log_message("CUDA available: " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        log_message("CUDA device name: " + torch.cuda.get_device_name(0))
    else:
        log_message("CUDA device not found. Please check your CUDA installation.")
    mp.set_start_method('spawn')
    train_model()
