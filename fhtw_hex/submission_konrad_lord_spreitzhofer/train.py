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
from multiprocessing import Pool, cpu_count, set_start_method

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def save_results(losses, win_rates, win_rates_checkpoint, model_folder):
    epochs = range(1, len(losses) + 1)

    # TODO plt checkpoint winrate
    print("checkpoint winrate: " + win_rates_checkpoint)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses, label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over Epochs')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, win_rates, label='Win Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Win Rate')
    plt.legend()
    plt.title('Win Rate over Epochs')
    plt.savefig(os.path.join(model_folder, 'loss_and_win_rate.png'))

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

def parallel_play_games(mcts, board_size, num_games, opponent='random'):
    with Pool(cpu_count()) as pool:
        results = pool.starmap(play_game, [(mcts, board_size, opponent) for _ in range(num_games)])
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

def validate_against_checkpoints(model, board_size, num_games=10, model_folder='models', checkpoints=[]):
    model.eval()
    current_mcts = MCTS(model)
    wins = 0

    with torch.no_grad():
        for checkpoint in checkpoints:
            # Load the checkpoint model
            checkpoint_model, _ = load_checkpoint(checkpoint, board_size)
            checkpoint_mcts = MCTS(checkpoint_model)

            # Play games between current model and checkpoint model
            for _ in range(num_games // len(checkpoints)):
                game = engine.HexPosition(board_size)
                while game.winner == 0:
                    if game.player == 1:
                        chosen = current_mcts.get_action(game.board, game.get_action_space())
                    else:
                        chosen = checkpoint_mcts.get_action(game.board, game.get_action_space())
                    game.moove(chosen)

                if game.winner == 1:
                    wins += 1

    win_rate = wins / num_games
    return win_rate


def train_model(board_size=config.BOARD_SIZE, epochs=config.EPOCHS, num_games_per_epoch=config.NUM_OF_GAMES_PER_EPOCH):
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
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}")
        opponent = 'random' if epoch < epochs // 2 else 'self'
        results = parallel_play_games(mcts, board_size, num_games_per_epoch, opponent=opponent)
         
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

        win_rate = validate_model(model, board_size, num_games=10)
        win_rates.append(win_rate)

        if epoch % 20 == 0:  # Save checkpoints every 10 epochs
            save_checkpoint(model, optimizer, epoch, model_folder, filename=f'checkpoint_epoch_{epoch}.pth.tar')
            checkpoints = [os.path.join(model_folder, f'checkpoint_epoch_{e}.pth.tar') for e in range(0, epoch+1, 20)]
            win_rate = validate_against_checkpoints(model, board_size, num_games=10, model_folder=model_folder, checkpoints=checkpoints)
            win_rates_checkpoint.append(win_rate)

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_model_state = model.state_dict()
            # torch.save(model.state_dict(), os.path.join(model_folder, 'best_hex_model.pth'))
            print("Saved best model with loss:", best_loss)

        print(f"Completed Epoch {epoch + 1}/{epochs} with loss: {loss.item()}, win rate: {win_rate}")
        


    torch.save(best_model_state, os.path.join(model_folder, 'best_hex_model.pth'))
    save_results(losses, win_rates, win_rates_checkpoint, model_folder)

if __name__ == "__main__":
    set_start_method('spawn')
    train_model()
