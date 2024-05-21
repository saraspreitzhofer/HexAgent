import numpy as np
import os
import copy
import matplotlib.pyplot as plt
import tensorflow as tf
from fhtw_hex.hex_engine import HexPosition
from facade import MCTS, create_model
from tqdm import tqdm
from datetime import datetime

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')


def select_device():
    """print("Select device for training:")
    print("1. GPU (if available)")
    print("2. CPU")
    choice = input("Enter 1 or 2: ")

    if choice == '1':
        if not tf.config.list_physical_devices('GPU'):
            print("No GPU found, using CPU instead.")
        else:
            print("Using GPU for training.")
    else:
        print("Using CPU for training.")
        tf.config.set_visible_devices([], 'GPU')"""

    if not tf.config.list_physical_devices('GPU'):
        print("No GPU found, using CPU instead.")
    else:
        print("Using CPU for training.")
        tf.config.set_visible_devices([], 'GPU')


def save_results(losses, win_rates, model_folder):
    epochs = range(1, len(losses) + 1)

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


def train_model(board_size=7, epochs=10, simulations=100, num_games_per_epoch=10):
    print("Creating model...")
    model = create_model(board_size)
    mcts = MCTS(model, simulations)
    best_loss = float('inf')

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

    for epoch in range(epochs):
        print(f"Starting Epoch {epoch + 1}/{epochs}")
        states, policies, values = [], [], []
        wins = 0

        for game_num in tqdm(range(num_games_per_epoch), desc=f'Epoch {epoch + 1}/{epochs}', unit='game'):
            game = HexPosition(board_size)
            state_history, policy_history, value_history = [], [], []

            while game.winner == 0:
                state_history.append(copy.deepcopy(game.board))
                action = mcts.get_action(game)
                game.moove(action)

            result = game.winner
            wins += 1 if result == 1 else 0

            for state in state_history:
                states.append(state)
                policies.append(np.random.dirichlet(np.ones(board_size * board_size)))
                values.append(result)

        states = np.array(states).reshape((-1, board_size, board_size, 1))
        policies = np.array(policies)
        values = np.array(values)

        print(f"Training model for Epoch {epoch + 1}/{epochs}")
        history = model.fit(states, [policies, values], epochs=1, verbose=0)

        loss = history.history['loss'][0]
        losses.append(loss)

        win_rate = wins / num_games_per_epoch
        win_rates.append(win_rate)

        if loss < best_loss:
            best_loss = loss
            model.save(os.path.join(model_folder, 'best_hex_model.keras'))
            print("Saved best model with loss:", best_loss)

        print(
            f"Completed Epoch {epoch + 1}/{epochs} with loss: {loss}, win rate: {win_rate}")

    save_results(losses, win_rates, model_folder)


if __name__ == "__main__":
    select_device()
    train_model(board_size=7, epochs=10, simulations=2, num_games_per_epoch=2)
