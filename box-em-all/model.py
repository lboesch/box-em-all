import numpy as np
import random

class QLearn:
    def __init__(self, alpha, gamma, epsilon):
        # Hyperparameters
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Initialize Q-table
        self.q_table = {}
        
    # Q-learning algorithm
    def update_q_table(self, game, state, action, reward):
        
        # for _ in range(1000):  # Run for a certain number of episodes
           # Q-learning formula
        old_q_value = self.q_table.get((state.tobytes(), action), 0)
        max_future_q = max([self.q_table.get((game.board.tobytes(), a), 0) for a in game.get_available_moves()], default=0)
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_q - old_q_value)
        self.q_table[(state.tobytes(), action)] = new_q_value