import numpy as np
import sys
import os
import copy
import matplotlib.pyplot as plt
from fhtw_hex import hex_engine as engine
from facade import MCTS, create_model
from tqdm import tqdm
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import config
import inspect
from torch.optim.lr_scheduler import StepLR
from multiprocessing import Pool
import torch.multiprocessing as mp
from random import choice
from collections import deque
import random
from fhtw_hex.submission_konrad_lord_spreitzhofer.utils import load_checkpoint, save_checkpoint, save_config_to_file, \
    save_results, setup_device

#
# Replay Buffer Klasse
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Collect logs during training
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
        state_history.append(copy.deepcopy(game.board))
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
    mcts = MCTS(model, epsilon=epsilon)
    return play_game(mcts, board_size, opponent)


def play_games(model, board_size, num_games, opponent='random', epsilon=config.EPSILON_START):
    model_state_dict = model.state_dict()
    total_cpus = os.cpu_count()
    if config.PARALLEL_GAMES:
        num_threads = min(total_cpus, config.NUM_PARALLEL_THREADS)
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
        mcts = MCTS(model, epsilon=epsilon)
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

    first_choice = True
    while game.winner == 0:
        if first_choice:
            if starter == "current":
                # Current agent (player1) macht den ersten Zug
                chosen = player1.get_action(game.board, game.get_action_space())
                first_choice = False
            else:
                # Checkpoint agent (player2) macht den ersten Zug zufällig
                chosen = choice(game.get_action_space())
                game.moove(chosen)
                first_choice = False
                continue  # Aktueller Agent (player1) ist an der Reihe
        elif game.player == 1:
            chosen = player1.get_action(game.board, game.get_action_space())
        else:
            if first_choice:
                chosen = choice(game.get_action_space())
                first_choice = False
            else:
                chosen = player2.get_action(game.board, game.get_action_space())
        game.moove(chosen)

    move_count = len(game.history)
    result = 1 if (
                (game.winner == 1 and starter == "current") or (game.winner == -1 and starter == "checkpoint")) else 0
    return result, move_count


def validate_against_checkpoints(model, board_size, num_games=config.NUM_OF_GAMES_PER_CHECKPOINT, model_folder='models',
                                 checkpoints=[]):
    model.eval()
    current_mcts = MCTS(model)
    win_rates = []
    move_rates = []

    with torch.no_grad():
        checkpoints_to_evaluate = checkpoints[:config.NUM_OF_AGENTS + 1]  # Only evaluate the desired number of checkpoints
        for i, checkpoint in enumerate(tqdm(checkpoints_to_evaluate, desc='Checkpoints', unit='checkpoint')):
            if 'random_agent_checkpoint.pth.tar' in checkpoint:
                checkpoint_mcts = RandomAgent()
            else:
                checkpoint_model, _ = load_checkpoint(checkpoint, board_size)
                checkpoint_mcts = MCTS(checkpoint_model)

            wins = 0
            total_moves = 0
            random_agent = True if i == 0 else False

            if config.PARALLEL_GAMES:
                total_cpus = os.cpu_count()
                num_threads = min(total_cpus, config.NUM_PARALLEL_THREADS)
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





def train_model(board_size=config.BOARD_SIZE, epochs=config.EPOCHS, num_games_per_epoch=config.NUM_OF_GAMES_PER_EPOCH):
    device = setup_device()
    log_message("Creating model...")
    model = create_model(board_size).to(device)
    mcts = MCTS(model)
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
    save_config_to_file(config, filename=os.path.join(model_folder, 'config.py'))

    losses = []
    policy_losses = []
    value_losses = []
    win_rates = [[] for _ in range(config.NUM_OF_AGENTS + 1)]  # Including Random Agent
    avg_moves = [[] for _ in range(config.NUM_OF_AGENTS + 1)]  # Including Random Agent

    criterion_policy = nn.KLDivLoss(reduction='batchmean')
    criterion_value = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.WARMUP_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, config.STEP_SIZE, gamma=config.GAMMA)
    model.to(device)

    replay_buffer = ReplayBuffer(capacity=10000)

    for epoch in range(1, epochs + 1):
        log_message(f"Starting Epoch {epoch}/{epochs}")
        if epoch <= config.WARMUP_EPOCHS:
            lr = config.WARMUP_LEARNING_RATE + (config.LEARNING_RATE - config.WARMUP_LEARNING_RATE) * (
                        epoch / config.WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif epoch == config.WARMUP_EPOCHS + 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.LEARNING_RATE

        epsilon = config.EPSILON_END + (config.EPSILON_START - config.EPSILON_END) * (1 - epoch / epochs)
        results = play_games(model, board_size, num_games_per_epoch,
                             opponent='self' if epoch > config.RANDOM_EPOCHS else 'random', epsilon=epsilon)

        for state_history, result in results:
            for state in state_history:
                policies = np.random.dirichlet(np.ones(board_size * board_size))
                replay_buffer.add((state, policies, result))

        if len(replay_buffer) < config.BATCH_SIZE:
            log_message("Not enough samples in replay buffer. Skipping training.")
            continue

        states, policies, values = zip(*replay_buffer.sample(config.BATCH_SIZE))

        states = np.array(states).reshape((-1, 1, board_size, board_size))
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
        loss = config.POLICY_LOSS_WEIGHT * policy_loss + config.VALUE_LOSS_WEIGHT * value_loss

        loss.backward()
        optimizer.step()
        scheduler.step()

        losses.append(loss.item())
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())

        if epoch == 1 or epoch % config.CHECKPOINT_INTERVAL == 0:
            checkpoint_epoch = epoch
            save_checkpoint(model, optimizer, checkpoint_epoch, model_folder,
                            filename=f'checkpoint_epoch_{checkpoint_epoch}.pth.tar')

        win_rates_checkpoint, avg_moves_checkpoint = [], []
        if epoch % config.EVALUATION_INTERVAL == 0 or epoch == epochs:
            checkpoints = [random_agent_checkpoint_path] + [os.path.join(model_folder, f'checkpoint_epoch_{e}.pth.tar')
                                                            for e in range(config.CHECKPOINT_INTERVAL, epoch + 1,
                                                                           config.CHECKPOINT_INTERVAL)]
            checkpoints = checkpoints[:config.NUM_OF_AGENTS + 1]  # Limit the number of checkpoints to evaluate
            log_message(f"Evaluating against checkpoints: {checkpoints}")
            win_rates_checkpoint, avg_moves_checkpoint = validate_against_checkpoints(model, board_size,
                                                                                      num_games=config.NUM_OF_GAMES_PER_CHECKPOINT,
                                                                                      model_folder=model_folder,
                                                                                      checkpoints=checkpoints)
            for i, (wr, am) in enumerate(zip(win_rates_checkpoint, avg_moves_checkpoint)):
                win_rates[i].append(wr)
                avg_moves[i].append(am)

        total_loss = policy_loss.item() + value_loss.item()
        log_message(
            f"Completed Epoch {epoch}/{epochs} with Loss: {total_loss}, Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")
        log_message(
            f"Random_Agent: Win Rates: {win_rates[0][-1] if win_rates[0] else 'N/A'}, Avg. Moves: {avg_moves[0][-1] if avg_moves[0] else 'N/A'}")
        for i in range(1, len(win_rates)):
            log_message(
                f"Agent_Checkpoint_Epoch_{i * config.CHECKPOINT_INTERVAL}: Win Rates: {win_rates[i][-1] if win_rates[i] else 'N/A'}, Avg. Moves: {avg_moves[i][-1] if avg_moves[i] else 'N/A'}")

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
            agent_epochs = list(range(config.EVALUATION_INTERVAL, epochs + 1, config.EVALUATION_INTERVAL))
        else:
            start_epoch = config.CHECKPOINT_INTERVAL * i
            agent_epochs = list(range(start_epoch, epochs + 1, config.EVALUATION_INTERVAL))

        plt.figure()
        plt.plot(agent_epochs, agent_win_rates, label=f'{checkpoints[i]} Win Rate')
        plt.xlabel('Epoch')
        plt.ylabel('Win Rate')
        plt.legend()
        plt.title('Win Rate over Checkpoints')
        plt.savefig(os.path.join(best_model_path, f'win_rate_{i}.png'))
        plt.close()

        plt.figure()
        plt.plot(agent_epochs, agent_avg_moves, label=f'{checkpoints[i]} Avg. Moves')
        plt.xlabel('Epoch')
        plt.ylabel('Moves')
        plt.legend()
        plt.title('Moves over Checkpoints')
        plt.savefig(os.path.join(best_model_path, f'avg_moves_{i}.png'))
        plt.close()


if __name__ == "__main__":
    log_message("CUDA available: " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        log_message("CUDA device name: " + torch.cuda.get_device_name(0))
    else:
        log_message("CUDA device not found. Please check your CUDA installation.")
    mp.set_start_method('spawn')
    train_model()
