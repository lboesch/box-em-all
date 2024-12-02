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
    def __init__(self, q_table={}):
        # Initialize Q-table
        self.q_table = q_table
        
    # Get Q-value
    def get_q_value(self, state, action):
        return self.q_table.get((state.tobytes(), action), 0)
    
    # Update Q-value    
    def update_q_value(self, state, action, q_value):
        self.q_table[(state.tobytes(), action)] = q_value
    
"""
DQN (Deep Q-network)
Link: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
    # Writing the object to a file using pickle
    def save(self, name):
        base_path = 'model'
        os.makedirs(base_path, exist_ok=True)
        filename = os.path.join(base_path, name + '.pt')
        with open(filename, 'wb') as file:
            torch.save(self, file)
    
    # Reading the object back from the file
    @staticmethod
    def load(name):
        base_path = 'model'
        filename = os.path.join(base_path, name + '.pt')
        with open(filename, 'rb') as file:
            return torch.load(file)

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