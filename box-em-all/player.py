from abc import ABC, abstractmethod
from collections import deque
import model
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim

class Player(ABC):
    def __init__(self, player_number: int, player_name: str):
        self.player_number = player_number
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
class Human(Player):
    def reset(self):
        super().reset()
    
    def play_turn(self, game):
        # Let the human player choose an action using input
        try:
            row, col = map(int, input("Enter row and column for your action (e.g., 0 1): ").split())
            return game.make_move(row, col)
        except ValueError:
            print("Invalid input. Enter two integers separated by a space.")
            return True

"""
Random Player
"""
class ComputerRandom(Player):
    def reset(self):
        super().reset()
    
    def play_turn(self, game):
        # Pick a random available action
        action = random.choice(game.get_available_actions())
        row, col = action
        return game.make_move(row, col)
        
"""
Greedy Player
"""
class ComputerGreedy(Player):
    def reset(self):
        super().reset()
    
    def play_turn(self, game):
        action = ()
        # Prioritize actions that complete a box
        for available_action in game.get_available_actions():
            row, col = available_action
            if game.is_valid_action(row, col):
                completed_boxes = len(game.check_boxes(row, col, sim=True))
                if completed_boxes > 0:
                    action = (row, col)
                    break

        # If no box-completing actions, pick a random available action
        if not action:
            action = random.choice(game.get_available_actions())
            
        row, col = action
        return game.make_move(row, col)

"""
Q-learning Agent
"""
class ComputerQLearning(Player):
    def __init__(self, player_number, player_name, model):
        super().__init__(player_number, player_name)
        self.model = model
        
    def reset(self):
        super().reset()
        self.total_reward = 0
        
    def review_game(self, game):
        pass  # TODO

    def play_turn(self, game):
        old_state = game.get_game_state()  # np.copy(game.get_game_state())
                                  
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.model.epsilon:
            # Exploration
            action = random.choice(game.get_available_actions())
        else:
            # Exploitation
            action = self.model.predict(game)
        
        row, col = action
        
        another_move = game.make_move(row, col)
        boxes_3_edges = game.check_boxes(row, col, edges=3)
        boxes_4_edges = game.check_boxes(row, col, edges=4)

        # Calculate reward
        reward = 0
        if another_move:
            # Box completed 
            reward += len(boxes_4_edges) + 0.2 * len(boxes_3_edges)
        else:
            # Giving advantage to opponent
            if len(boxes_3_edges) > 0:
                reward -= len(boxes_3_edges)
            # Drawing edge without completing a box
            else:
                reward -= 0.1
        if game.is_game_over():  # TODO currently not possible to get a reward if the opponent has finished the game (MDP --> next state after opponents action?)
            # Winning a game
            if self.player_score > game.opponent_player.player_score:
                reward += 5
            # Loosing a game
            elif self.player_score <= game.opponent_player.player_score:
                reward -= 5
        self.total_reward += reward

        # Update Q-table
        self.model.update(game, old_state, action, reward, another_move)  # TODO deque?

        return another_move
    
"""
Q-table Player
"""
class ComputerQTable(Player):
    def __init__(self, player_number, player_name, model):
        super().__init__(player_number, player_name)
        self.model = model
        
    def reset(self):
        super().reset()

    def play_turn(self, game):
        action = self.model.predict(game)
        row, col = action    
        another_move = game.make_move(row, col)
        return another_move
    
"""
DQN Agent
"""
class ComputerDQN(Player):
    def __init__(self, player_number, player_name, model):
        super().__init__(player_number, player_name)
        self.memory = deque(maxlen=2000)
        #
        self.gamma = 0.99
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.learning_rate = 0.001
        self.model = model
        #
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.loss_funct = nn.MSELoss()
        
    def reset(self):
        super().reset()
        self.total_reward = 0
        
    def play_turn(self, game):
        state = game.get_game_state()
        action = self.next_action(game)
        row, col = action
        another_move = game.make_move(row, col)
        next_state = game.get_game_state()
        reward = 1 # TODO
        self.total_reward += reward
        done = True # TODO
        self.memory.append((state, game.get_idx_by_action(row, col), reward, next_state, done))
        return another_move
    
    def next_action(self, game):
        if np.random.rand() <= self.epsilon:
            return random.choice(game.get_available_actions()) # TODO
        state = torch.tensor(game.get_game_state(), dtype=torch.float32)
        q_values = self.model(state)
        return game.get_action_by_idx(torch.argmax(q_values).item())  # TODO transform action to scalar value
    
    # Replay with previous experience to train model
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return
        minibatch = random.sample(self.memory, batch_size)
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
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay