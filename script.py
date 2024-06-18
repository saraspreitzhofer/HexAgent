# importing the module
from fhtw_hex import hex_engine as engine
# this is how your agent can be imported
# 'submission_konrad_lord_spreitzhofer' is the (sub)package that you provide
from fhtw_hex.submission_konrad_lord_spreitzhofer.facade import agent
from fhtw_hex.submission_konrad_lord_spreitzhofer.facade import MCTS, create_model
import torch
import os
from fhtw_hex.submission_konrad_lord_spreitzhofer import config

# initializing a game object
game = engine.HexPosition()

# Load the trained model
model = create_model(config.BOARD_SIZE)
model.load_state_dict(torch.load(config.MODEL, map_location=torch.device('cpu')))

# Initialize MCTS with the trained model
mcts_agent = MCTS(model)

# Function to use the trained agent for moves
def trained_agent(board, action_set):
    return mcts_agent.get_action(board, action_set)

# Uncomment this if you want to enable


# let your agent play against random
print("Agent against random")
game.machine_vs_machine(machine1=trained_agent, machine2=None)
print("Random against agent")
game.machine_vs_machine(machine1=None, machine2=trained_agent)

# let your agent play against itself
print("Agent against agent")
game.machine_vs_machine(machine1=trained_agent, machine2=trained_agent)

print("Human against agent")
game.human_vs_machine(machine=trained_agent)
