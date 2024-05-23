from copy import deepcopy
import numpy as np  
from fhtw_hex import hex_engine as engine
from fhtw_hex.submission_konrad_lord_spreitzhofer import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
class Node:
    hex_position_class = engine.HexPosition

    def __init__(self, state, action_space=None, parent=None, action=None, prior=0):
        self.state = state
        self.action_space = action_space
        self.parent = parent
        self.action = action
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior
        self.hex_position = self.create_hex_position(state)

    def create_hex_position(self, state):
        temp_hex_position = self.hex_position_class(size=len(state))
        temp_hex_position.board = state
        temp_hex_position.player = 1 if sum(sum(row) for row in state) == 0 else -1  # Determine the player
        return temp_hex_position

    def is_terminal(self):
        return self.hex_position.winner != 0

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, policy):
        valid_actions = self.hex_position.get_action_space()  # Ensure only valid moves are considered
        for action, prob in zip(self.action_space, policy):
            if action in valid_actions:  # Check if the action is valid
                new_state = deepcopy(self.state)
                temp_hex_position = self.create_hex_position(new_state)
                temp_hex_position.moove(action)
                self.children.append(
                    Node(temp_hex_position.board, parent=self, action=action, prior=prob,
                         action_space=self.action_space)
                )

    def select_child(self):
        return max(self.children, key=lambda x: x.value())

    def update(self, value):
        self.visit_count += 1
        self.value_sum += value

    def value(self):
        epsilon = 1e-6  # Small value to prevent division by zero
        return self.value_sum / (self.visit_count + epsilon) + self.prior


class MCTS:
    def __init__(self, model, simulations=100, device=device):
        self.model = model
        self.simulations = simulations
        self.device = device
        self.model.to(self.device)

    def get_action(self, state, action_set):
        root = Node(state, action_set)
        for _ in range(self.simulations):
            self.search(root)
        return max(root.children, key=lambda c: c.visit_count).action

    def search(self, node):
        if node.is_terminal():
            return -node.state.winner
        if not node.is_expanded():
            policy, value = self.evaluate(node)
            node.expand(policy)
            return -value
        child = node.select_child()
        value = self.search(child)
        node.update(-value)
        return -value

    def evaluate(self, node):
        board = np.array(node.state).reshape((1, 1, config.BOARD_SIZE, config.BOARD_SIZE)).astype(np.float32)
        board = torch.tensor(board, device=self.device)
        self.model.eval()
        with torch.no_grad():
            policy, value = self.model(board)
        return policy.cpu().numpy()[0], value.cpu().numpy()[0][0]



import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class HexNet(nn.Module):
    def __init__(self, board_size):
        super(HexNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.flatten = nn.Flatten()
        self.policy_head = nn.Linear(64 * board_size * board_size, board_size * board_size)
        self.value_head = nn.Linear(64 * board_size * board_size, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.flatten(x)
        policy = F.softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value

def create_model(board_size):
    model = HexNet(board_size).to(device)  # Move model to GPU
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, optimizer



# Here should be the necessary Python wrapper for your model, in the form of a callable agent, such as above.
# Please make sure that the agent does actually work with the provided Hex module.

def agent(board, action_set):
    board_size = config.BOARD_SIZE
    model, optimizer = create_model(board_size)
    model.load_state_dict(torch.load(config.MODEL, map_location=device))  # Load the model weights
    return MCTS(model).get_action(board, action_set)
