from abc import ABC, abstractmethod
from collections import namedtuple, deque
from datetime import datetime
from game import DotsAndBoxes
import os
import pickle
import random
import torch
import torch.nn as nn

"""
Policy Base Class
"""
class Policy(ABC):
    @abstractmethod
    def next_action(self, game):
        pass
    
    @staticmethod
    def model_name(name):
        return name + '_' + datetime.now().strftime("%Y%m%d%H%M%S")
    
    # Writing the object to a file
    def save(self, name):
        base_path = 'model'
        os.makedirs(base_path, exist_ok=True)
        filename = os.path.join(base_path, name + '.pkl')
        with open(filename, 'wb') as file:
            pickle.dump(self, file)
    
    # Reading the object from a file
    @staticmethod
    def load(name):
        base_path = 'model'
        filename = os.path.join(base_path, name + '.pkl')
        with open(filename, 'rb') as file:
            return pickle.load(file)
        
# ====================================================================================================
# Random
# ====================================================================================================
class Random(Policy):  
    # Predict next action
    def next_action(self, game):
        return random.choice(game.get_available_actions())

# ====================================================================================================
# Greedy
# ====================================================================================================
class Greedy(Policy):
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
class QLearning(Policy):
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
        return max(game.get_random_available_actions(), key=lambda action: self.get_q_value(game.board, game.get_idx_by_action(*action)))
    
# ====================================================================================================
# DQN
# ====================================================================================================
class DQNBase(nn.Module, Policy):
    def __init__(self, board_size):
        super().__init__()
        self.board_size = board_size
        self.state_size = self.action_size = DotsAndBoxes.calc_game_state_size(board_size)

    # Get device
    def get_device(self):
        return next(self.parameters()).device
    
    # Get state
    @abstractmethod
    def get_state(self):
        pass
    
    # Writing the object to a file
    def save(self, name):
        base_path = 'model'
        os.makedirs(base_path, exist_ok=True)
        filename = os.path.join(base_path, name + '.pt')
        with open(filename, 'wb') as file:
            torch.save(self, file)
    
    # Reading the object from a file
    @staticmethod
    def load(name):
        base_path = 'model'
        filename = os.path.join(base_path, name + '.pt')
        with open(filename, 'rb') as file:
            return torch.load(file, weights_only=False)

'''
DQN without convolutional layers
'''
class DQN(DQNBase):
    def __init__(self, board_size):
        super().__init__(board_size)
        self.input_shape = (self.state_size,)  # state size
        self.net = nn.Sequential(
            nn.Linear(self.input_shape[0], 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)
        )
        
    def forward(self, x):
        return self.net(x)
    
    def get_state(self, board):
        return DotsAndBoxes.get_game_state(board)
    
    # Predict next action
    def next_action(self, game):
        with torch.no_grad():
            state = torch.tensor(self.get_state(game.board), dtype=torch.float, device=self.get_device())
            q_values = self(state)
            # return game.get_action_by_idx(torch.argmax(q_values).item())  # TODO use instead of the following line?
            return max(game.get_random_available_actions(), key=lambda action: q_values[game.get_idx_by_action(*action)].item())

'''
DQN with convolutional layers
'''
class DQNConv(DQNBase):
    def __init__(self, board_size):
        super().__init__(board_size)
        # self.input_shape = (2, 2 * board_size + 1, 2 * board_size + 1)  # channels, height, width
        self.input_shape = (2, board_size + 1, board_size + 1)  # channels, height, width
        # Convolutional layers
        self.conv_layers = nn.Sequential(
            nn.Conv2d(self.input_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Flatten()
        )
        self.fc_input_size = self._get_conv_output_size()  # Calculate the size of the input to the first fully connected layer
        # Dueling Network: Fully connected layers
        self.advantage_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.action_size)  # Q-values for all actions
        )
        self.value_layers = nn.Sequential(
            nn.Linear(self.fc_input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        
    def _get_conv_output_size(self):
        input = torch.rand(1, *self.input_shape)
        output = self.conv_layers(input)
        output_size = output.data.view(1, -1).size(1)
        return output_size

    def forward(self, x):
        # x = x.view(x.shape[0], self.input_shape[0], x.shape[2], x.shape[3])  # Batch size, channels, height, width
        feature = self.conv_layers(x)
        # Dueling Network
        value = self.value_layers(feature)
        advantage = self.advantage_layers(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q
    
    def get_state(self, board):
        return DotsAndBoxes.get_game_state_2d_2ch(board)
    
    # Predict next action
    def next_action(self, game):
        with torch.no_grad():
            state = torch.tensor(self.get_state(game.board), dtype=torch.float, device=self.get_device())
            q_values = torch.squeeze(self(state.unsqueeze(0)))
            return max(game.get_random_available_actions(), key=lambda action: q_values[game.get_idx_by_action(*action)].item())