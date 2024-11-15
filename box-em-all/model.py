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
    def __init__(self, alpha, gamma, epsilon, q_table={}):
        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        # Initialize Q-table
        self.q_table = q_table
        
    # Q-learning algorithm
    def update_q_table(self, game, old_state, action, reward, another_move):
        is_fututre_move = 1 if another_move else -1
        # Q-learning formula
        old_q_value = self.q_table.get((old_state, action), 0)
        state, index = game.get_canonical_state_and_rotation()
        moves = game.get_canonical_moves(index);

        max_future_q = max([self.q_table.get((state, a), 0) for a in moves], default=0)
        update_value = reward + is_fututre_move * self.gamma * max_future_q
        new_q_value = old_q_value + self.alpha * (update_value - old_q_value)
        self.q_table[(old_state, action)] = new_q_value