import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import torch.nn.functional as F
from copy import deepcopy
from fhtw_hex import hex_engine as engine
from fhtw_hex.submission_konrad_lord_spreitzhofer import config as base_config
from fhtw_hex.submission_konrad_lord_spreitzhofer.facade import create_model, MCTS, RandomAgent, log_message, save_log_to_file, SumTree, PrioritizedReplayBuffer
from fhtw_hex.submission_konrad_lord_spreitzhofer.utils import load_checkpoint, save_checkpoint, save_config_to_file, save_results, setup_device
import os
from datetime import datetime
from tqdm import tqdm
from multiprocessing import Pool
import torch.multiprocessing as mp
from random import choice

# Convert config module to a dictionary(this should enable to train on cluster parallel)
base_config_dict = {key: getattr(base_config, key) for key in dir(base_config) if not key.startswith("__")}
config = deepcopy(base_config_dict)

def play_game(mcts: MCTS, board_size: int, opponent='self', move_penalty=0.01):
    game = engine.HexPosition(board_size)
    state_history = []

    game.player = choice([1, -1])  # Randomize the starting player
    move_count = 0

    while game.winner == 0:
        state_history.append((deepcopy(game.board), game.player))
        if game.player == 1:
            chosen = mcts.get_action(game.board, game.get_action_space())
        else:
            chosen = mcts.get_action(game.board, game.get_action_space()) if opponent == 'self' else choice(game.get_action_space())
        game.moove(chosen)
        move_count += 1
        if game.winner != 0:
            break

    # Calculate rewards
    max_moves = board_size * board_size
    normalized_move_penalty = move_penalty / max_moves

    if game.winner == 1:
        player1_reward = 1 - (normalized_move_penalty * move_count)
        player2_reward = -(1 - (normalized_move_penalty * move_count))
    elif game.winner == -1:
        player1_reward = -(1 - (normalized_move_penalty * move_count))
        player2_reward = 1 - (normalized_move_penalty * move_count)

    # Separate state histories for both players
    player1_states = [(state, player) for state, player in state_history if player == 1]
    player2_states = [(state, player) for state, player in state_history if player == -1]

    return player1_states, player2_states, player1_reward, player2_reward

def play_game_worker(args):
    model_state_dict, board_size, opponent, device, epsilon, temperature = args
    model = create_model(board_size).to(device)
    model.load_state_dict(model_state_dict)
    mcts = MCTS(model, epsilon=epsilon, temperature=temperature, board_size=board_size)
    return play_game(mcts, board_size, opponent)

def play_games(model, board_size, num_games, opponent='random', epsilon=0.1, temperature=1.0):
    model_state_dict = model.state_dict()
    total_cpus = os.cpu_count()
    if config['PARALLEL_GAMES']:
        num_threads = min(total_cpus, config['NUM_PARALLEL_THREADS'])
        log_message(f"Using {num_threads} parallel threads out of {total_cpus} available")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mp.set_start_method('spawn', force=True)
        with Pool(num_threads) as pool:
            args = [(model_state_dict, board_size, opponent, device, epsilon, temperature) for _ in range(num_games)]
            results = list(tqdm(pool.imap(play_game_worker, args), total=num_games, unit='game'))
        return results
    else:
        log_message(f"Parallel games disabled. Using a single thread out of {total_cpus} available")
        results = []
        mcts = MCTS(model, epsilon=epsilon, temperature=temperature, board_size=board_size)
        for _ in tqdm(range(num_games), unit='game'):
            results.append(play_game(mcts, board_size, opponent))
        return results

def play_validation(args):
    board_size, current_mcts, checkpoint_mcts, random_agent = args
    game = engine.HexPosition(board_size)
    starter = choice(["current", "checkpoint"])  # Zufällige Auswahl des Startspielers

    if random_agent:
        checkpoint_mcts = RandomAgent()
    player1, player2 = (current_mcts, checkpoint_mcts) if starter == "current" else (checkpoint_mcts, current_mcts)

    first_move_done_by_player2 = False

    while game.winner == 0:
        if game.player == 1:
            if player1 == checkpoint_mcts and not first_move_done_by_player2:
                chosen = choice(game.get_action_space())
                first_move_done_by_player2 = True
            else:
                chosen = player1.get_action(game.board, game.get_action_space())
        else:
            if player2 == checkpoint_mcts and not first_move_done_by_player2:
                chosen = choice(game.get_action_space())
                first_move_done_by_player2 = True
            else:
                chosen = player2.get_action(game.board, game.get_action_space())
        game.moove(chosen)

        if game.winner != 0:
            break  # Ensure no more moves are made after the game ends

    move_count = len(game.history) - 1
    result = 1 if ((game.winner == 1 and starter == "current") or (game.winner == -1 and starter == "checkpoint")) else 0
    return result, move_count

def validate_against_checkpoints(model, board_size, num_games=10, model_folder='models', checkpoints=[]):
    model.eval()
    current_mcts = MCTS(model, board_size=board_size)
    win_rates = []
    move_rates = []

    checkpoints = checkpoints[:config['NUM_OF_AGENTS'] + 1]

    with torch.no_grad():
        for i, checkpoint in enumerate(tqdm(checkpoints, desc='Checkpoints', unit='checkpoint')):
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

            avg_moves = total_moves / num_games
            if avg_moves > board_size * board_size:
                log_message(f"Warning: Average moves ({avg_moves}) exceed the maximum possible moves ({board_size * board_size}). There might be an issue.")
            win_rates.append(wins / num_games)
            move_rates.append(avg_moves)

    return win_rates, move_rates

def compute_td_error(state, policy, value, model, device):
    state_tensor = torch.tensor(np.array(state).reshape((1, 1, config['BOARD_SIZE'], config['BOARD_SIZE'])), dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        policy_pred, value_pred = model(state_tensor)
    policy_pred = policy_pred.cpu().numpy().flatten()
    value_pred = value_pred.cpu().numpy().flatten()[0]
    td_error = np.abs(value - value_pred) + np.sum(np.abs(policy - policy_pred))
    return td_error

def train_model():
    local_config = deepcopy(config)

    device = setup_device()
    log_message("Creating model...")
    model = create_model(local_config['BOARD_SIZE']).to(device)
    mcts = MCTS(model, simulations=local_config['MCTS_SIMULATIONS'], epsilon=local_config['EPSILON_START'], temperature=local_config['TEMPERATURE_START'], board_size=local_config['BOARD_SIZE'])
    best_loss = float('inf')
    best_model_state = None

    models_folder = 'models'
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_folder = os.path.join(models_folder, current_time)
    os.makedirs(model_folder, exist_ok=True)

    random_agent_checkpoint_path = os.path.join(model_folder, 'random_agent_checkpoint.pth.tar')
    torch.save({'state_dict': None, 'optimizer': None}, random_agent_checkpoint_path)

    log_message("Saving config to file...")
    save_config_to_file(local_config, filename=os.path.join(model_folder, 'config.py'))

    losses = []
    policy_losses = []
    value_losses = []
    win_rates = [[] for _ in range(local_config['NUM_OF_AGENTS'] + 1)]
    avg_moves = [[] for _ in range(local_config['NUM_OF_AGENTS'] + 1)]

    criterion_policy = nn.KLDivLoss(reduction='batchmean')
    criterion_value = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=local_config['WARMUP_LEARNING_RATE'], weight_decay=local_config['WEIGHT_DECAY'])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    model.to(device)

    replay_buffer = PrioritizedReplayBuffer(capacity=local_config['REPLAY_BUFFER_CAPACITY'])

    for epoch in range(1, local_config['EPOCHS'] + 1):
        log_message(f"Starting Epoch {epoch}/{local_config['EPOCHS']}")
        if epoch <= local_config['WARMUP_EPOCHS']:
            lr = local_config['WARMUP_LEARNING_RATE'] + (
                    local_config['LEARNING_RATE'] - local_config['WARMUP_LEARNING_RATE']) * (
                         epoch / local_config['WARMUP_EPOCHS'])
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif epoch == local_config['WARMUP_EPOCHS'] + 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = local_config['LEARNING_RATE']

        # Adjust epsilon
        epsilon_decay_rate = np.log(local_config['EPSILON_END'] / local_config['EPSILON_START']) / local_config['EPOCHS']
        epsilon = local_config['EPSILON_START'] * np.exp(epsilon_decay_rate * epoch)

        # Adjust temperature
        temperature_decay_rate = np.log(local_config['TEMPERATURE_END'] / local_config['TEMPERATURE_START']) / local_config['EPOCHS']
        temperature = local_config['TEMPERATURE_START'] * np.exp(temperature_decay_rate * epoch)

        results = play_games(model, local_config['BOARD_SIZE'], local_config['NUM_OF_GAMES_PER_EPOCH'], opponent='self', epsilon=epsilon, temperature=temperature)

        for player1_states, player2_states, player1_reward, player2_reward in results:
            for state, player in player1_states:
                policies = np.random.dirichlet(np.ones(local_config['BOARD_SIZE'] * local_config['BOARD_SIZE']))
                reward = player1_reward
                td_error = compute_td_error(state, policies, reward, model, device)
                replay_buffer.add((state, policies, reward), td_error)
            for state, player in player2_states:
                policies = np.random.dirichlet(np.ones(local_config['BOARD_SIZE'] * local_config['BOARD_SIZE']))
                reward = player2_reward
                td_error = compute_td_error(state, policies, reward, model, device)
                replay_buffer.add((state, policies, reward), td_error)

        if len(replay_buffer) < local_config['BATCH_SIZE']:
            log_message("Not enough samples in replay buffer. Skipping training.")
            continue

        indices, experiences, is_weights = replay_buffer.sample(local_config['BATCH_SIZE'])
        states, policies, rewards = zip(*experiences)

        states = np.array(states).reshape((-1, 1, local_config['BOARD_SIZE'], local_config['BOARD_SIZE']))
        policies = np.array(policies)
        rewards = np.array(rewards).astype(np.float32)

        states = torch.tensor(states, dtype=torch.float32).to(device)
        policies = torch.tensor(policies, dtype=torch.float32).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        is_weights = torch.tensor(is_weights, dtype=torch.float32).to(device)

        model.train()
        optimizer.zero_grad()

        policy_outputs, value_outputs = model(states)
        policy_loss = criterion_policy(policy_outputs, policies)
        value_loss = criterion_value(value_outputs.squeeze(), rewards)
        loss = local_config['POLICY_LOSS_WEIGHT'] * policy_loss + local_config['VALUE_LOSS_WEIGHT'] * value_loss

        weighted_loss = loss * is_weights.mean()
        weighted_loss.backward()
        optimizer.step()
        scheduler.step(loss)

        losses.append(loss.item())
        policy_losses.append(policy_loss.item())
        value_losses.append(value_loss.item())

        if epoch == 1 or epoch % local_config['CHECKPOINT_INTERVAL'] == 0:
            checkpoint_epoch = epoch
            save_checkpoint(model, optimizer, checkpoint_epoch, model_folder, filename=f'checkpoint_epoch_{checkpoint_epoch}.pth.tar')

        win_rates_checkpoint, avg_moves_checkpoint = [], []
        if epoch % local_config['EVALUATION_INTERVAL'] == 0 or epoch == local_config['EPOCHS']:
            checkpoints = [random_agent_checkpoint_path] + [os.path.join(model_folder, f'checkpoint_epoch_{e}.pth.tar') for e in range(local_config['CHECKPOINT_INTERVAL'], epoch + 1, local_config['CHECKPOINT_INTERVAL'])]
            checkpoints = checkpoints[:local_config['NUM_OF_AGENTS'] + 1]
            log_message(f"Evaluating against checkpoints: {checkpoints}")
            win_rates_checkpoint, avg_moves_checkpoint = validate_against_checkpoints(model, local_config['BOARD_SIZE'], num_games=local_config['NUM_OF_GAMES_PER_CHECKPOINT'], model_folder=model_folder, checkpoints=checkpoints)
            for i, (wr, am) in enumerate(zip(win_rates_checkpoint, avg_moves_checkpoint)):
                win_rates[i].append(wr)
                avg_moves[i].append(am)

        total_loss = policy_loss.item() + value_loss.item()
        log_message(f"Completed Epoch {epoch}/{local_config['EPOCHS']} with Loss: {total_loss}, Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")
        log_message(f"Random_Agent: Win Rates: {win_rates[0][-1] if win_rates[0] else 'N/A'}, Avg. Moves: {avg_moves[0][-1] if avg_moves[0] else 'N/A'}")
        for i in range(1, len(win_rates)):
            log_message(f"Agent_Checkpoint_Epoch_{i * local_config['CHECKPOINT_INTERVAL']}: Win Rates: {win_rates[i][-1] if win_rates[i] else 'N/A'}, Avg. Moves: {avg_moves[i][-1] if avg_moves[i] else 'N/A'}")

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()

    best_model_path = os.path.join(model_folder, 'best_loss')
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    torch.save(best_model_state, os.path.join(best_model_path, 'best_hex_model.pth'))
    save_results(losses, win_rates, policy_losses, value_losses, best_model_path, avg_moves, checkpoints, local_config)

    log_path = os.path.join(model_folder, 'train.log')
    save_log_to_file(log_path)
    log_message(f"Logfile created at {log_path}")

if __name__ == "__main__":
    log_message("CUDA available: " + str(torch.cuda.is_available()))
    if torch.cuda.is_available():
        log_message("CUDA device name: " + torch.cuda.get_device_name(0))
    else:
        log_message("CUDA device not found. Please check your CUDA installation.")
    mp.set_start_method('spawn')
    train_model()
