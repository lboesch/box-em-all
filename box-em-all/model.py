import numpy as np
import os
import pickle

# Writing the object to a file using pickle
def save(name, model):
    base_path = 'model'
    os.makedirs(base_path, exist_ok=True)
    filename = os.path.join(base_path, name + '.pkl')
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    # np.save(filename, model)
 
# Reading the object back from the file
def load(name):
    base_path = 'model'
    filename = os.path.join(base_path, name + '.pkl')
    with open(filename, 'rb') as file:
        return pickle.load(file)
    # return np.load(filename, allow_pickle=True).item()

# Q-learning
class QLearning:
    def __init__(self, alpha, gamma, epsilon):
        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        # Initialize Q-table
        self.q_table = {}
        
    # Q-learning algorithm
    def update_q_table(self, game, old_state, action, reward):
        # Q-learning formula
        old_q_value = self.q_table.get((old_state.tobytes(), action), 0)
        max_future_q = max([self.q_table.get((game.board.tobytes(), a), 0) for a in game.available_moves], default=0)
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_q - old_q_value)
        self.q_table[(old_state.tobytes(), action)] = new_q_value