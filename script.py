import torch
import os
from fhtw_hex import hex_engine as engine
from fhtw_hex.submission_konrad_lord_spreitzhofer.facade import agent, MCTS, create_model


def load_trained_agent(board_size):
    model_path = f'fhtw_hex/submission_konrad_lord_spreitzhofer/models/hex{board_size}x{board_size}/best_loss/best_hex_model.pth'

    if not os.path.exists(model_path):
        print(f"Sorry, boardsize {board_size}x{board_size} is not available")
        return None

    model = create_model(board_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    mcts_agent = MCTS(model)
    return mcts_agent


def trained_agent(board, action_set, mcts_agent):
    return mcts_agent.get_action(board, action_set)


if __name__ == "__main__":
    available_boards = [3, 4, 5, 7]
    board_size = int(input("Which boardsize do you want to play? Press 3, 4, 5 or 7: "))

    if board_size not in available_boards:
        print(f"Sorry, boardsize {board_size} is not available")
    else:
        mcts_agent = load_trained_agent(board_size)

        if mcts_agent:
            game = engine.HexPosition(board_size)


            # Function to use the trained agent for moves
            def agent_wrapper(board, action_set):
                return trained_agent(board, action_set, mcts_agent)


            print("Agent against random")
            game.machine_vs_machine(machine1=agent_wrapper, machine2=None)
            print("Random against agent")
            game.machine_vs_machine(machine1=None, machine2=agent_wrapper)

            print("Agent against agent")
            game.machine_vs_machine(machine1=agent_wrapper, machine2=agent_wrapper)

            print("Human against agent")
            game.human_vs_machine(machine=agent_wrapper)
