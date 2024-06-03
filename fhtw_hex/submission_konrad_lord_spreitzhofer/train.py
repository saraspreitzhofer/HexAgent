import numpy as np
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
from multiprocessing import Pool, cpu_count
import itertools
import torch.multiprocessing as mp


# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def save_config_to_file(config_module, filename="config.py"):
    with open(filename, 'w') as file:
        for name, value in inspect.getmembers(config_module):
            if not name.startswith("__") and not inspect.ismodule(value) and not inspect.isfunction(value):
                file.write(f"{name} = {value}\n")

def save_results(losses, win_rates, policy_losses, value_losses, model_folder):
    epochs = range(1, len(losses) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    for i, win_rate in enumerate(win_rates):
        start_epoch = i * config.CHECKPOINT_INTERVAL + 1
        plt.plot(range(start_epoch, start_epoch + len(win_rate)), win_rate, label=f'Checkpoint {start_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.title('Win Rate over Checkpoints')
    plt.savefig(os.path.join(model_folder, 'loss_and_win_rate.png'))
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(policy_losses) + 1), policy_losses, label='Policy Loss')
    plt.plot(range(1, len(value_losses) + 1), value_losses, label='Value Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Policy and Value Losses over Epochs')
    plt.savefig(os.path.join(model_folder, 'policy_and_value_losses.png'))
    plt.close()

def play_game(mcts: MCTS, board_size: int, opponent='random'):
    game = engine.HexPosition(board_size)
    state_history = []

    while game.winner == 0:
        state_history.append(copy.deepcopy(game.board))
        if game.player == 1:
            chosen = mcts.get_action(game.board, game.get_action_space())
        else:
            if opponent == 'random':
                from random import choice
                chosen = choice(game.get_action_space())
            elif opponent == 'self':
                chosen = mcts.get_action(game.board, game.get_action_space())
        game.moove(chosen)

    result = game.winner
    return state_history, result

def play_game_worker(args):
    model_state_dict, board_size, opponent, device = args
    model = create_model(board_size).to(device)
    model.load_state_dict(model_state_dict)
    mcts = MCTS(model)
    result = play_game(mcts, board_size, opponent)
    return result

def play_games(model, board_size, num_games, opponent='random'):
    model_state_dict = model.state_dict()
    total_cpus = os.cpu_count()  # Anzahl der verfügbaren CPUs
    if config.PARALLEL_GAMES:
        num_threads = min(total_cpus, config.NUM_PARALLEL_THREADS)
        print(f"Using {num_threads} parallel threads out of {total_cpus} available")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        mp.set_start_method('spawn', force=True)
        with Pool(num_threads) as pool:
            args = [(model_state_dict, board_size, opponent, device) for _ in range(num_games)]
            results = list(tqdm(pool.imap(play_game_worker, args), total=num_games, unit='game'))
        return results
    else:
        print(f"Parallel games disabled. Using a single thread out of {total_cpus} available")
        results = []
        mcts = MCTS(model)
        for _ in tqdm(range(num_games), unit='game'):
            results.append(play_game(mcts, board_size, opponent))
        return results

def save_checkpoint(model, optimizer, epoch, model_folder, filename='checkpoint.pth.tar'):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    filepath = os.path.join(model_folder, filename)
    torch.save(state, filepath)

def load_checkpoint(filepath, board_size):
    model = create_model(board_size)
    optimizer = optim.Adam(model.parameters())
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer


def play_validation(args):
    board_size, current_mcts, checkpoint_mcts = args
    game = engine.HexPosition(board_size)
    while game.winner == 0:
        if game.player == 1:
            chosen = current_mcts.get_action(game.board, game.get_action_space())
        else:
            chosen = checkpoint_mcts.get_action(game.board, game.get_action_space())
        game.moove(chosen)

        return 1 if game.winner == 1 else 0

def validate_against_checkpoints(model, board_size, num_games=config.NUM_OF_GAMES_PER_CHECKPOINT, model_folder='models', checkpoints=[]):
    model.eval()
    current_mcts = MCTS(model)
    win_rates = []

    with torch.no_grad():
        for checkpoint in tqdm(checkpoints, desc='Checkpoints', unit='checkpoint'):
            checkpoint_model, _ = load_checkpoint(checkpoint, board_size)
            checkpoint_mcts = MCTS(checkpoint_model)
            wins = 0

            if config.PARALLEL_GAMES:
                total_cpus = os.cpu_count()  # Anzahl der verfügbaren CPUs
                num_threads = min(total_cpus, config.NUM_PARALLEL_THREADS)
                mp.set_start_method('spawn', force=True)
                with Pool(num_threads) as pool:
                    args = [(board_size, current_mcts, checkpoint_mcts) for _ in range(num_games)]
                    results = list(tqdm(pool.imap(play_validation, args), total=num_games, unit='game'))
                    wins = sum(results)
            else:
                for _ in range(num_games):
                    wins += play_validation(board_size, current_mcts, checkpoint_mcts)
            print(f"Win rate against {checkpoint}: {wins} / {num_games}")
            win_rates.append(wins / num_games)

    return win_rates

def setup_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        device = torch.device('cpu')
        print("CUDA device not found. Using CPU.")
    return device

def train_model(board_size=config.BOARD_SIZE, epochs=config.EPOCHS, num_games_per_epoch=config.NUM_OF_GAMES_PER_EPOCH):
    device = setup_device()
    print("Creating model...")
    model = create_model(board_size).to(device)
    mcts = MCTS(model)
    best_loss = float('inf')
    best_model_state = None

    models_folder = 'models'
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_folder = os.path.join(models_folder, current_time)
    os.makedirs(model_folder)

    print("Saving config to file...")
    save_config_to_file(config, filename=os.path.join(model_folder, 'config.py'))

    losses = []
    policy_losses = []
    value_losses = []
    win_rates = []
    win_rates_checkpoint = []

    criterion_policy = nn.KLDivLoss(reduction='batchmean')
    criterion_value = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.WARMUP_LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = StepLR(optimizer, config.STEP_SIZE, gamma=config.GAMMA)
    model.to(device)

    for epoch in range(1, epochs + 1):
        print(f"Starting Epoch {epoch}/{epochs}")
        if epoch <= config.WARMUP_EPOCHS:
            lr = config.WARMUP_LEARNING_RATE + (config.LEARNING_RATE - config.WARMUP_LEARNING_RATE) * (epoch / config.WARMUP_EPOCHS)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        elif epoch == config.WARMUP_EPOCHS + 1:
            for param_group in optimizer.param_groups:
                param_group['lr'] = config.LEARNING_RATE

        if epoch == config.RANDOM_EPOCHS:
            print("Switching to self-play")
        opponent = 'random' if epoch < config.RANDOM_EPOCHS else 'self'
        results = play_games(model, board_size, num_games_per_epoch, opponent=opponent)

        states, policies, values = [], [], []
        wins = 0

        for state_history, result in results:
            wins += 1 if result == 1 else 0
            for state in state_history:
                states.append(state)
                policies.append(np.random.dirichlet(np.ones(board_size * board_size)))
                values.append(result)

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
            checkpoint_epoch = 0 if epoch == 1 else epoch
            save_checkpoint(model, optimizer, checkpoint_epoch, model_folder, filename=f'checkpoint_epoch_{checkpoint_epoch}.pth.tar')
            win_rates.append([])
            # if checkpoint_epoch >= (config.CHECKPOINT_INTERVAL * config.NUM_OF_OPPONENTS_PER_CHECKPOINT):
            #     checkpoints = [os.path.join(model_folder, f'checkpoint_epoch_{e}.pth.tar') for e in range((epoch) - (config.NUM_OF_OPPONENTS_PER_CHECKPOINT * config.CHECKPOINT_INTERVAL), epoch - config.CHECKPOINT_INTERVAL + 1, config.CHECKPOINT_INTERVAL)]
            #     win_rate = validate_against_checkpoints(model, board_size, num_games=config.NUM_OF_GAMES_PER_CHECKPOINT, model_folder=model_folder, checkpoints=checkpoints)
            #     win_rates_checkpoint.append(win_rate)

        win_rates_checkpoint = validate_against_checkpoints(model, board_size, num_games=config.NUM_OF_GAMES_PER_CHECKPOINT, model_folder=model_folder, checkpoints=[os.path.join(model_folder, f'checkpoint_epoch_{e}.pth.tar') for e in range(0, epoch + 1, config.CHECKPOINT_INTERVAL)])
        for i, wr in enumerate(win_rates_checkpoint):
            win_rates[i].append(wr)

        print(win_rates)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()

        print(f"Completed Epoch {epoch}/{epochs} with loss: {loss.item()}")
        print(f"Policy Loss: {policy_loss.item()}, Value Loss: {value_loss.item()}")

    best_model_path = os.path.join(model_folder, 'best_loss')
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    torch.save(best_model_state, os.path.join(best_model_path, 'best_hex_model.pth'))
    save_results(losses, win_rates, policy_losses, value_losses, best_model_path)



if __name__ == "__main__":
    print("CUDA available: ", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device name: ", torch.cuda.get_device_name(0))
    else:
        print("CUDA device not found. Please check your CUDA installation.")
    mp.set_start_method('spawn')
    train_model()
