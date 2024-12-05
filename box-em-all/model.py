from abc import ABC, abstractmethod
from collections import namedtuple, deque
from datetime import datetime
import numpy as np
import os
import pickle
import random
import torch
import torch.nn as nn
import torch.optim as optim

"""
Model Base Class
"""
class Model(ABC):
    @abstractmethod
    def next_action(self, game):
        pass
    
    # Writing the object to a file using pickle
    def save(self, name):
        base_path = 'model'
        os.makedirs(base_path, exist_ok=True)
        filename = os.path.join(base_path, name + '_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.pkl')
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    
    # Reading the object back from the file
    @staticmethod
    def load(name):
        base_path = 'model'
        filename = os.path.join(base_path, name + '.pkl')
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
# ====================================================================================================
# Random
# ====================================================================================================
class Random(Model):  
    # Predict next action
    def next_action(self, game):
        return random.choice(game.get_available_actions())

# ====================================================================================================
# Greedy
# ====================================================================================================
class Greedy(Model):
    # Predict next action
    def next_action(self, game):
        action = ()
        # Prioritize actions that complete a box
        for available_action in game.get_available_actions():
            boxes = game.check_boxes(*available_action, sim=True)
            if len(boxes[4]) > 0:
                action = available_action
                break
        # If no box-completing actions, pick a random available action
        if not action:
            action = random.choice(game.get_available_actions())
        return action

# ====================================================================================================
# Q-learning
# ====================================================================================================
class QLearning(Model):
    def __init__(self, q_table={}):
        # Initialize Q-table
        self.q_table = q_table
        
    # Get Q-value
    def get_q_value(self, state, action):
        return self.q_table.get((state.tobytes(), action), 0)
    
    # Update Q-value    
    def update_q_value(self, state, action, q_value):
        self.q_table[(state.tobytes(), action)] = q_value
        
    # Predict next action
    def next_action(self, game):
        return max(game.get_random_available_actions(), key=lambda action: self.get_q_value(game.get_game_state(), action))  # TODO what if multiple max?
    
# ====================================================================================================
# DQN
# ====================================================================================================
# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
# ====================================================================================================
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
    
    def get_device(self):
        return next(self.parameters()).device
    
    # Predict next action
    def next_action(self, game):
        state = torch.tensor(game.get_game_state(), dtype=torch.float32, device=self.get_device())
        q_values = self(state)
        # return game.get_action_by_idx(torch.argmax(q_values).item())  # TODO transform action to scalar value
        return max(game.get_random_available_actions(), key=lambda action: q_values[game.get_idx_by_action(*action)].item())
    
    # Writing the object to a file using pickle
    def save(self, name):
        base_path = 'model'
        os.makedirs(base_path, exist_ok=True)
        filename = os.path.join(base_path, name + '_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.pt')
        with open(filename, 'wb') as file:
            torch.save(self, file)
    
    # Reading the object back from the file
    @staticmethod
    def load(name):
        base_path = 'model'
        filename = os.path.join(base_path, name + '.pt')
        with open(filename, 'rb') as file:
            return torch.load(file, weights_only=False)

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