from abc import ABC, abstractmethod
from collections import deque
import model
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim


"""
Player Base Class
"""
class Player(ABC):
    def __init__(self, player_name: str):
        self.player_number = None
        self.player_name = player_name
        self.reset()

    @abstractmethod
    def reset(self):
        self.player_score = 0  # TODO move to game?

    @abstractmethod
    def play_turn(self, game):
        pass
 
    def review_game(self, game):
        pass
    
"""
Human Player
"""
class HumanPlayer(Player):
    def reset(self):
        super().reset()
    
    def play_turn(self, game):
        # Let the human player choose an action using input
        try:
            row, col = map(int, input("Enter row and column for your action (e.g., 0 1): ").split())
            another_move, _ = game.make_move(row, col)
            return another_move
        except ValueError:
            print("Invalid input. Enter two integers separated by a space.")
            return True

"""
Random Player
"""
class RandomPlayer(Player):
    def reset(self):
        super().reset()
    
    def play_turn(self, game):
        # Pick a random available action
        action = random.choice(game.get_available_actions())
        row, col = action
        another_move, _ = game.make_move(row, col)
        return another_move
        
"""
Greedy Player
"""
class GreedyPlayer(Player):
    def reset(self):
        super().reset()
    
    def play_turn(self, game):
        action = ()
        # Prioritize actions that complete a box
        for available_action in game.get_available_actions():
            row, col = available_action
            if game.is_valid_action(row, col):  # TODO remove?
                boxes = game.check_boxes(row, col, sim=True)
                if len(boxes[4]) > 0:
                    action = (row, col)
                    break

        # If no box-completing actions, pick a random available action
        if not action:
            action = random.choice(game.get_available_actions())
            
        row, col = action
        another_move, _ = game.make_move(row, col)
        return another_move

# ====================================================================================================
# Q-learning
# ====================================================================================================
class QLearning(Player):
    # Predict next action
    def next_action(self, game):
        actions = game.get_available_actions()
        random.shuffle(actions)
        return max(actions, key=lambda action: self.model.get_q_value(game.get_game_state(), action))  # TODO what if multiple max?

"""
Q-learning Agent
"""
class QAgent(QLearning):
    def __init__(self, player_name, model, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
        super().__init__(player_name)
        self.model = model
        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
    def reset(self):
        super().reset()
        self.total_reward = 0
        self.state = None
        self.action = None

    def play_turn(self, game):
        self.state = game.get_game_state()
                                  
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            # Exploration
            self.action = random.choice(game.get_available_actions())
        else:
            # Exploitation
            self.action = self.next_action(game)
        
        row, col = self.action
        another_move, boxes = game.make_move(row, col)
        
        # Calculate reward
        reward = game.calc_reward(another_move, boxes)         
        self.total_reward += reward

        # Update Q-table
        self.update_q_table(self.state, self.action, reward, game)  # TODO deque?
        
        return another_move
    
    def review_game(self, game):
        reward = 0
        # Winning a game
        if self.player_score > game.opponent_player.player_score:
            reward += 1
        # Loosing a game
        elif self.player_score < game.opponent_player.player_score:
            reward -= 1
            
        self.update_q_table(self.state, self.action, reward, game)
    
    # Update epsilon (decrease exploration)
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # Update Q-table (Q-learning algorithm)
    def update_q_table(self, state, action, reward, game):
        old_q_value = self.model.get_q_value(state, action)
        max_future_q = max([self.model.get_q_value(game.get_game_state(), action) for action in game.get_available_actions()], default=0)
        # Formula: Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_q - old_q_value)
        self.model.update_q_value(state, action, new_q_value)
    
"""
Q-learning Player
"""
class QPlayer(QLearning):
    def __init__(self, player_name, model):
        super().__init__(player_name)
        self.model = model
        
    def reset(self):
        super().reset()

    def play_turn(self, game):
        action = self.next_action(game)
        row, col = action
        another_move, _ = game.make_move(row, col)
        return another_move
                
# ====================================================================================================
# DQN
# ====================================================================================================
class DQN(Player):
    # Predict next action
    def next_action(self, game):
        state = torch.tensor(game.get_game_state(), dtype=torch.float32)
        q_values = self.model(state)
        # return game.get_action_by_idx(torch.argmax(q_values).item())  # TODO transform action to scalar value
        return max(game.get_available_actions(), key=lambda action: q_values[game.get_idx_by_action(*action)].item())

"""
DQN Agent
"""
class DQNAgent(DQN):
    def __init__(self, player_name, model, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
        super().__init__(player_name)
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        self.loss_funct = nn.MSELoss()
        self.memory = deque(maxlen=2000)
        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
    def reset(self):
        super().reset()
        self.total_reward = 0
        
    def play_turn(self, game):
        state = game.get_game_state()
        if np.random.rand() <= self.epsilon:
            action = random.choice(game.get_available_actions()) # TODO
        else:
            action = self.next_action(game)
        row, col = action
        another_move, boxes = game.make_move(row, col)
        next_state = game.get_game_state()
        reward = game.calc_reward(another_move, boxes)
        self.total_reward += reward
        done = True # TODO
        self.memory.append((state, game.get_idx_by_action(row, col), reward, next_state, done))
        return another_move      
    
    # Replay with previous experience to train model
    def optimize(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
        # Optimize net
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(torch.tensor(next_state, dtype=torch.float32))) # TODO
            target_f = self.model(torch.tensor(state, dtype=torch.float32)).detach().numpy()
            target_f[action] = target
            self.model.zero_grad()
            loss = self.loss_funct(self.model(torch.tensor(state, dtype=torch.float32)), torch.tensor(target_f))
            loss.backward()
            self.optimizer.step()
        # Decrease epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
"""
DQN Player
"""
class DQNPlayer(DQN):
    def __init__(self, player_name, model):
        super().__init__(player_name)
        self.model = model
        
    def reset(self):
        super().reset()
        
    def play_turn(self, game):
        action = self.next_action(game)
        row, col = action
        another_move, _ = game.make_move(row, col)
        return another_move    