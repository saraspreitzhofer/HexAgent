# importing the module
from fhtw_hex import hex_engine as engine
# this is how your agent can be imported
# 'submission_konrad_lord_spreitzhofer' is the (sub)package that you provide
from fhtw_hex.submission_konrad_lord_spreitzhofer.facade import agent
from fhtw_hex.submission_konrad_lord_spreitzhofer.facade import MCTS
import os
from fhtw_hex.submission_konrad_lord_spreitzhofer import config




# initializing a game object
game = engine.HexPosition()

# make sure that the agent you have provided is such that the following three
# method-calls are error-free and as expected


#Uncomment this if you want to enable

# let your agent play against random
'''
print("Agent against random")
game.machine_vs_machine(machine1=agent, machine2=None)
print("Random against agent")
game.machine_vs_machine(machine1=None, machine2=agent)

# let your agent play against itself
print("Agent against agent")
game.machine_vs_machine(machine1=agent, machine2=agent)
'''


print("Human against agent")
game.human_vs_machine()
