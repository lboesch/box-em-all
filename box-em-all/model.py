import numpy as np
import os
import pickle

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

# Q-learning
class QLearning(Model):
    def __init__(self, alpha, gamma, epsilon, q_table={}):
        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        # Initialize Q-table
        self.q_table = q_table
        
    # Q-learning algorithm
    def update_q_table(self, game, old_state, action, reward, another_move):
        is_future_move = 1 if another_move else -1
        # Q-learning formula
        old_q_value = self.q_table.get((old_state.tobytes(), action), 0)
        max_future_q = max([self.q_table.get((game.get_game_state().tobytes(), move), 0) for move in game.available_moves], default=0)
        new_q_value = old_q_value + self.alpha * (reward + is_future_move * self.gamma * max_future_q - old_q_value)
        self.q_table[(old_state.tobytes(), action)] = new_q_value