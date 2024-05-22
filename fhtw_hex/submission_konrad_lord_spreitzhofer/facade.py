from copy import deepcopy
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras import models
from fhtw_hex import hex_engine as engine
from fhtw_hex.submission_konrad_lord_spreitzhofer import config


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
    def __init__(self, model, simulations=100):
        self.model = model
        self.simulations = simulations

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
        board = np.array(node.state).reshape((1, config.BOARD_SIZE, config.BOARD_SIZE, 1))
        policy, value = self.model.predict(board, verbose=0)
        return policy[0], value[0][0]


def create_model(board_size):
    inputs = Input(shape=(board_size, board_size, 1))
    x = Conv2D(32, (3, 3), padding='same', activation='relu')(inputs)
    x = Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = Flatten()(x)
    policy = Dense(board_size * board_size, activation='softmax', name='policy_output')(x)
    value = Dense(1, activation='tanh', name='value_output')(x)
    model = models.Model(inputs=inputs, outputs=[policy, value])
    model.compile(loss=['categorical_crossentropy', 'mean_squared_error'], optimizer=Adam(learning_rate=0.001))
    return model


# Here should be the necessary Python wrapper for your model, in the form of a callable agent, such as above.
# Please make sure that the agent does actually work with the provided Hex module.

def agent(board, action_set):
    model = models.load_model(config.MODEL)
    return MCTS(model).get_action(board, action_set)
