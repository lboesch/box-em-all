from collections import namedtuple, deque
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim

class Model:
    # Writing the object to a file using pickle
    def save(self, name):
        base_path = 'model'
        os.makedirs(base_path, exist_ok=True)
        filename = os.path.join(base_path, name + '.pkl')
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    
    # Reading the object back from the file
    @staticmethod
    def load(name):
        base_path = 'model'
        filename = os.path.join(base_path, name + '.pkl')
        with open(filename, 'rb') as file:
            return pickle.load(file)

"""
Q-learning
"""
class QLearning(Model):
    def __init__(self, alpha, gamma, epsilon, q_table={}):
        # Hyperparameters
        self.alpha = alpha  # Learning rate TODO move to agent?
        self.gamma = gamma  # Discount factor TODO move to agent?
        self.epsilon = epsilon  # Exploration rate TODO move to agent?
        # Initialize Q-table
        self.q_table = q_table
        
    # Update Q-table
    def update(self, game, old_state, action, reward, another_move):
        is_future_move = 1 if another_move else -1 # TODO necessary?
        # Q-learning algorithm
        old_q_value = self.q_table.get((old_state.tobytes(), action), 0)
        max_future_q = max([self.q_table.get((game.get_game_state().tobytes(), action), 0) for action in game.get_available_actions()], default=0)
        # Formula: Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        new_q_value = old_q_value + self.alpha * (reward + is_future_move * self.gamma * max_future_q - old_q_value)
        self.q_table[(old_state.tobytes(), action)] = new_q_value
    
    # Predict next action
    def predict(self, game):
        return max(game.get_available_actions(), key=lambda action: self.q_table.get((game.get_game_state().tobytes(), action), 0))
    
"""
DQN (Deep Q-network)
Link: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Named tuple for transitions
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# Replay memory
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)