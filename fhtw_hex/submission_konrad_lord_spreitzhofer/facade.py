import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy
from fhtw_hex import hex_engine as engine
from fhtw_hex.submission_konrad_lord_spreitzhofer import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

    def select_child(self, exploration_weight=1.0):
        return max(self.children, key=lambda x: x.value(exploration_weight))

    def update(self, value):
        self.visit_count += 1
        self.value_sum += value

    def value(self, exploration_weight=1.0):
        epsilon = 1e-6
        if self.visit_count == 0:
            return float('inf')
        exploitation = self.value_sum / (self.visit_count + epsilon)
        exploration = exploration_weight * self.prior * np.sqrt(
            np.log(self.parent.visit_count + 1) / (self.visit_count + epsilon))
        return exploitation + exploration


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out


class HexNet(nn.Module):
    def __init__(self, board_size):
        super(HexNet, self).__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        self.residual_blocks = nn.ModuleList([ResidualBlock(64) for _ in range(3)]) #Anzahl der Blöce

        self.flatten = nn.Flatten()
        self.policy_head = nn.Linear(64 * board_size * board_size, board_size * board_size)
        self.value_head = nn.Linear(64 * board_size * board_size, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))

        for block in self.residual_blocks:
            x = block(x)

        x = self.flatten(x)
        policy = F.log_softmax(self.policy_head(x), dim=1)
        value = torch.tanh(self.value_head(x))
        return policy, value


def create_model(board_size):
    model = HexNet(board_size).to(device)
    return model


class MCTS:
    def __init__(self, model, simulations=config.MCTS_SIMULATIONS, device=device, epsilon=config.EPSILON_START):
        self.model = model
        self.simulations = simulations
        self.device = device
        self.epsilon = epsilon
        self.model.to(self.device)

    def get_action(self, state, action_set):
        root = Node(state, action_set)
        for _ in range(self.simulations):
            self.search(root)

        if np.random.rand() < self.epsilon:
            return np.random.choice(root.children).action

        return max(root.children, key=lambda c: c.visit_count).action

    def search(self, node, exploration_weight=1.0):
        if node.is_terminal():
            return -node.state.winner
        if not node.is_expanded():
            policy, value = self.evaluate(node)
            node.expand(policy)
            return -value
        child = node.select_child(exploration_weight)
        value = self.search(child, exploration_weight)
        node.update(-value)
        return -value

    def evaluate(self, node):
        board = np.array(node.state).reshape((1, 1, config.BOARD_SIZE, config.BOARD_SIZE)).astype(np.float32)
        board = torch.tensor(board, device=self.device)
        self.model.eval()
        with torch.no_grad():
            policy, value = self.model(board)

        # Apply temperature to policy
        policy = np.exp(policy.cpu().numpy()[0] / config.TEMPERATURE)
        policy = policy / np.sum(policy)  # Normalize

        return policy, value.cpu().numpy()[0][0]


def agent(board, action_set):
    board_size = config.BOARD_SIZE
    model = create_model(board_size)
    model.load_state_dict(torch.load(config.MODEL, map_location=device))
    model.to(device)  # Move the model to GPU
    return MCTS(model).get_action(board, action_set)
