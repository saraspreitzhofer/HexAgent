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
from multiprocessing import Pool, cpu_count, set_start_method, Manager
import inspect

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def save_config_to_file(config_module, filename="config.py"):
    with open(filename, 'w') as file:
        for name, value in inspect.getmembers(config_module):
            if not name.startswith("__") and not inspect.ismodule(value) and not inspect.isfunction(value):
                file.write(f"{name} = {value}\n")

def save_results(losses, win_rates, win_rates_checkpoint, model_folder):
    epochs = range(1, len(losses) + 1)

    print(f"checkpoint winrate: [{', '.join(str(element) for element in win_rates_checkpoint)}]")
    config.save_config_to_json(config.config, filename=os.path.join(model_folder, 'config.json'))

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(win_rates_checkpoint)+1), win_rates_checkpoint, label='Win Rate')
    plt.xlabel('Checkpoint')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.title('Win Rate over Checkpoints')
    plt.savefig(os.path.join(model_folder, 'loss_and_win_rate.png'))

    plt.close()

class Counter:
    def __init__(self, total):
        self.value = 0
        self.total = total
        self.lock = Manager().Lock()

    def increment(self):
        with self.lock:
            self.value += 1

    def progress(self):
        with self.lock:
            return self.value / self.total

def play_game(mcts: MCTS, board_size: int, opponent='random', counter: Counter = None):
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
    if counter:
        counter.increment()
    return state_history, result

def play_games(mcts, board_size, num_games, opponent='random', parallel=False):
    if parallel:
        manager = Manager()
        counter = Counter(num_games)

        def update_progress():
            with tqdm(total=num_games, unit="game") as pbar:
                while True:
                    pbar.n = counter.value
                    pbar.refresh()
                    if counter.value >= num_games:
                        break

        pool = Pool(cpu_count())
        watcher = pool.apply_async(update_progress)
        results = pool.starmap(play_game, [(mcts, board_size, opponent, counter) for _ in range(num_games)])
        pool.close()
        pool.join()
        watcher.wait()
        return results
    else:
        results = []
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


def validate_model(model, board_size, num_games=10):
    model.eval()
    mcts = MCTS(model)
    wins = 0

    with torch.no_grad():
        for _ in range(num_games):
            game = engine.HexPosition(board_size)
            while game.winner == 0:
                if game.player == 1:
                    chosen = mcts.get_action(game.board, game.get_action_space())
                else:
                    chosen = mcts.get_action(game.board, game.get_action_space())
                game.moove(chosen)

            if game.winner == 1:
                wins += 1

    win_rate = wins / num_games
    return win_rate

def validate_against_checkpoints(model, board_size, num_games=config.NUM_OF_GAMES_PER_CHECKPOINT, model_folder='models', checkpoints=[]):
    model.eval()
    current_mcts = MCTS(model)
    wins = 0

    with torch.no_grad():
        for checkpoint in tqdm(checkpoints, desc='Checkpoints', unit='checkpoint'):
            # Load the checkpoint model
            checkpoint_model, _ = load_checkpoint(checkpoint, board_size)
            checkpoint_mcts = MCTS(checkpoint_model)

            # Play games between current model and checkpoint model
            for _ in range(num_games):
                game = engine.HexPosition(board_size)
                while game.winner == 0:
                    if game.player == 1:
                        chosen = current_mcts.get_action(game.board, game.get_action_space())
                    else:
                        chosen = checkpoint_mcts.get_action(game.board, game.get_action_space())
                    game.moove(chosen)

                if game.winner == 1:
                    wins += 1

    win_rate = wins / (num_games * len(checkpoints))
    return win_rate

def setup_device():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    return device

def train_model(board_size=config.BOARD_SIZE, epochs=config.EPOCHS, num_games_per_epoch=config.NUM_OF_GAMES_PER_EPOCH):
    device = setup_device()
    print("Saving config to file...")
    save_config_to_file(config, filename='config.py')
    print("Creating model...")
    model = create_model(board_size)
    mcts = MCTS(model)
    best_loss = float('inf')
    best_model_state = None

    # Create the 'models' folder if it doesn't exist
    models_folder = 'models'
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)

    # Create a new folder for this training session
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    model_folder = os.path.join(models_folder, current_time)
    os.makedirs(model_folder)

    losses = []
    win_rates = []
    win_rates_checkpoint = []

    criterion_policy = nn.CrossEntropyLoss()
    criterion_value = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    model.to(device)

    for epoch in range(1, epochs+1):
        print(f"Starting Epoch {epoch}/{epochs}")
        if epoch == epochs / 2:
            print("Switching to self-play")
        opponent = 'random' if epoch < epochs // 2 else 'self'
        results = play_games(mcts, board_size, num_games_per_epoch, opponent=opponent, parallel=True)
         
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
        loss = policy_loss + value_loss

        loss.backward()
        optimizer.step()

        losses.append(loss.item())

        win_rate = None

        if epoch == 1 or epoch % config.CHECKPOINT_INTERVAL == 0:  # Save checkpoints and validate
            checkpoint_epoch = 0 if epoch == 1 else epoch
            save_checkpoint(model, optimizer, checkpoint_epoch, model_folder, filename=f'checkpoint_epoch_{checkpoint_epoch}.pth.tar')
            # only validate if we have enough checkpoints to compare against
            if checkpoint_epoch >= (config.CHECKPOINT_INTERVAL * config.NUM_OF_OPPONENTS_PER_CHECKPOINT):
                # Validate against the last few checkpoints
                checkpoints = [os.path.join(model_folder, f'checkpoint_epoch_{e}.pth.tar') for e in range((epoch) - (config.NUM_OF_OPPONENTS_PER_CHECKPOINT * config.CHECKPOINT_INTERVAL), epoch-config.CHECKPOINT_INTERVAL+1, config.CHECKPOINT_INTERVAL)]
                win_rate = validate_against_checkpoints(model, board_size, num_games=config.NUM_OF_GAMES_PER_CHECKPOINT, model_folder=model_folder, checkpoints=checkpoints)
                win_rates_checkpoint.append(win_rate)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()

        print(f"Completed Epoch {epoch}/{epochs} with loss: {loss.item()}")
        if win_rate:
            print(f"Win rate: {win_rate}")

    best_model_path = os.path.join(model_folder, 'best_loss')
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    torch.save(best_model_state, os.path.join(best_model_path, 'best_hex_model.pth'))
    save_results(losses, win_rates, win_rates_checkpoint, best_model_path)

if __name__ == "__main__":
    set_start_method('spawn')
    train_model()
