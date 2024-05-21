# importing the module
from fhtw_hex import hex_engine as engine
# this is how your agent can be imported
# 'submission_konrad_lord_spreitzhofer' is the (sub)package that you provide
from fhtw_hex.submission_konrad_lord_spreitzhofer.facade import agent
from fhtw_hex.submission_konrad_lord_spreitzhofer.facade import MCTS
import os
from tensorflow.keras.models import load_model


def load_model_from_folder(model_folder):
    model_path = os.path.join(model_folder, 'best_hex_model.keras')
    return load_model(model_path)


def display_board(game):
    game.print(invert_colors=True)
    print(f"Current player: {'White' if game.player == 1 else 'Black'}")
    print(f"Current winner: {game.winner}")


def human_vs_agent(board_size=7, simulations=100, model_folder=None):
    game = engine.HexPosition(board_size)
    model = load_model_from_folder(model_folder)
    mcts = MCTS(model, simulations)

    while game.winner == 0:
        display_board(game)
        if game.player == 1:  # Human player
            valid_move = False
            while not valid_move:
                try:
                    move = input("Enter your move (row,col): ")
                    row, col = move.split(',')
                    row = int(row) - 1
                    col = ord(col.upper()) - ord('A')
                    if (row, col) in game.get_action_space():
                        game.moove((row, col))
                        valid_move = True
                    else:
                        print("Invalid move. Try again.")
                except ValueError:
                    print("Invalid input format. Use (row,col). Try again.")
        else:  # Agent's move
            action = mcts.get_action(game)
            print(f"Agent move: {chr(action[1] + ord('A'))}{action[0] + 1}")
            game.moove(action)

    display_board(game)
    print(f"{'White' if game.winner == 1 else 'Black'} wins!")


model_dir = 'models'
model_folders = [f for f in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, f))]
if not model_folders:
    print("No models found. Please train a model first.")
else:
    print("Select a model to play against:")
    for i, folder in enumerate(model_folders):
        print(f"{i + 1}. {folder}")
    choice = int(input("Enter the number of the model you want to select: ")) - 1
    selected_model_folder = os.path.join(model_dir, model_folders[choice])

    human_vs_agent(simulations=1, model_folder=selected_model_folder)


# initializing a game object
game = engine.HexPosition()

# make sure that the agent you have provided is such that the following three
# method-calls are error-free and as expected

# let your agent play against random
game.machine_vs_machine(machine1=agent, machine2=None)
game.machine_vs_machine(machine1=None, machine2=agent)

# let your agent play against itself
game.machine_vs_machine(machine1=agent, machine2=agent)
