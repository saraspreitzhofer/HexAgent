# stupid example
def agent(board, action_set):
    return action_set[0]    # todo!

# Here should be the necessary Python wrapper for your model, in the form of a callable agent, such as above.
# Please make sure that the agent does actually work with the provided Hex module.

from copy import deepcopy
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten
from tensorflow.keras import models

class Node:
    def __init__(self, state, parent=None, action=None, prior=0):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.visit_count = 0
        self.value_sum = 0
        self.prior = prior

    def is_terminal(self):
        return self.state.winner != 0

    def is_expanded(self):
        return len(self.children) > 0

    def expand(self, policy):
        actions = self.state.get_action_space()
        for action, prob in zip(actions, policy):
            new_state = deepcopy(self.state)
            new_state.moove(action)
            self.children.append(Node(new_state, parent=self, action=action, prior=prob))

    def select_child(self):
        return max(self.children, key=lambda x: x.value())

    def update(self, value):
        self.visit_count += 1
        self.value_sum += value

    def value(self):
        epsilon = 1e-6  # Small value to prevent division by zero
        return self.value_sum / (self.visit_count + epsilon) + self.prior

class MCTS:
    def __init__(self, model, simulations):
        self.model = model
        self.simulations = simulations

    def get_action(self, state):
        root = Node(state)
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
        board = np.array(node.state.board).reshape((1, node.state.size, node.state.size, 1))
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
