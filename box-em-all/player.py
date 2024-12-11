from abc import ABC, abstractmethod
from collections import deque
from game import DotsAndBoxes
import copy
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
    def act(self, game):
        pass
 
    def review_game(self, game):
        pass
    
# class Agent(Player):
#     def __init__(self, player_name, policy):
#         super().__init__(player_name)
#         self.policy = policy
    
#     def reset(self):
#         super().reset()
    
#     def act(self, game):
#         action = self.policy.next_action(game)
#         another_step, _ = game.step(*action)
#         return another_step
    
"""
Human Player
"""
class HumanPlayer(Player):
    def reset(self):
        super().reset()
    
    def act(self, game):
        # Let the human player choose an action using input
        try:
            action = map(int, input("Enter row and column for your action (e.g., 0 1): ").split())
            another_step, _ = game.step(*action)
            return another_step
        except ValueError:
            print("Invalid input. Enter two integers separated by a space.")
            return True

"""
Random Player
"""
class RandomPlayer(Player):
    def __init__(self, player_name):
        super().__init__(player_name)
        self.model = model.Random()
    
    def reset(self):
        super().reset()
    
    def act(self, game):
        action = self.model.next_action(game)
        another_step, _ = game.step(*action)
        return another_step
        
"""
Greedy Player
"""
class GreedyPlayer(Player):
    def __init__(self, player_name):
        super().__init__(player_name)
        self.model = model.Greedy()
    
    def reset(self):
        super().reset()
    
    def act(self, game):
        action = action = self.model.next_action(game)
        another_step, _ = game.step(*action)
        return another_step

# ====================================================================================================
# Q-learning
# ====================================================================================================
class QLearning(Player):
    def __init__(self, player_name, model):
        super().__init__(player_name)
        self.model = model

"""
Q-learning Agent
"""
class QAgent(QLearning):
    def __init__(self, player_name, model, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
        super().__init__(player_name, model)
        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_update_freq = 100
        #
        self.steps = 0
        
    def reset(self):
        super().reset()
        self.state = None
        self.action = None
        self.boxes = None
        self.another_step = None
        self.reward = 0
        self.total_reward = 0

    def act(self, game):
        if self.state is not None:  # TODO hack -> deque, dataclass, SimpleNamespace?
            self.reward = game.calc_reward(self.boxes, self.another_step)
            self.total_reward += self.reward
            # Update Q-table
            self.update_q_table(self.state, self.action, self.reward, game)
        if not game.is_game_over():
            self.state = game.get_game_state()                 
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < self.epsilon:
                # Exploration
                self.action = random.choice(game.get_available_actions())
            else:
                # Exploitation
                self.action = self.model.next_action(game)
            self.another_step, self.boxes = game.step(*self.action)
            self.steps += 1
            # Update epsilon
            if self.steps % self.epsilon_update_freq == 0:
                self.update_epsilon()
            return self.another_step
        else:
            return False
    
    def review_game(self, game):
        self.act(game)
    
    # Update epsilon (decrease exploration)
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # Update Q-table (Q-learning algorithm)
    def update_q_table(self, state, action, reward, game):
        old_q_value = self.model.get_q_value(state, action)
        max_future_q = max([self.model.get_q_value(game.get_game_state(), action) for action in game.get_available_actions()], default=0)
        # Formula: Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        new_q_value = old_q_value + self.alpha * (reward + self.gamma * max_future_q - old_q_value)  # TODO
        self.model.update_q_value(state, action, new_q_value)
    
"""
Q-learning Player
"""
class QPlayer(QLearning):
    def reset(self):
        super().reset()

    def act(self, game):
        action = self.model.next_action(game)
        another_step, _ = game.step(*action)
        return another_step
                
# ====================================================================================================
# DQN
# ====================================================================================================
class DQN(Player):
    def __init__(self, player_name, model):
        super().__init__(player_name)
        self.model = model
        self.device = model.get_device()

"""
DQN Agent
"""
class DQNAgent(DQN):
    def __init__(self, player_name, model, alpha, gamma, epsilon, epsilon_decay, epsilon_min):
        super().__init__(player_name, model)
        # self.target_net = model.DQN(state_size, action_size)
        # self.target_net.load_state_dict(model.state_dict())
        self.target_model = copy.deepcopy(model)
        self.target_model.to(self.device)
        self.target_model.eval()
        # self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        # self.optimizer = optim.AdamW(self.model.parameters(), lr=alpha)
        self.optimizer = optim.SGD(self.model.parameters(), lr=alpha)
        self.loss_funct = nn.MSELoss()
        # self.loss_funct = nn.SmoothL1Loss()
        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon_update_freq = 100
        # Training
        self.batch_size = 128
        self.max_replay_size = 32 * self.batch_size
        self.min_replay_size = 8 * self.batch_size
        self.replay_memory = deque(maxlen=self.max_replay_size)
        self.model_update_freq = 4
        self.target_network_update_freq = 100
        #
        self.steps = 0
        
    def reset(self):
        super().reset()
        self.state = None
        self.action = None
        self.boxes = None
        self.another_step = None
        self.score_diff = 0
        self.reward = 0
        self.total_reward = 0
        self.last_loss = 0
        
    def act(self, game):
        if self.state is not None:  # TODO hack -> deque, dataclass, SimpleNamespace?
            self.reward = game.calc_reward(self.boxes, self.another_step, self.score_diff)
            self.total_reward += self.reward
            # Update replay memory: State, Action, Reward, Next State, Game Over
            self.replay_memory.append((self.state, game.get_idx_by_action(*self.action), self.reward, game.get_game_state(), game.is_game_over()))
            # Learn
            if self.steps % self.model_update_freq == 0 or game.is_game_over: # or done:
                self.last_loss = self.learn()
            # Update target net
            if self.steps % self.target_network_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())
        if not game.is_game_over():
            self.state = game.get_game_state()
            if np.random.rand() <= self.epsilon:
                self.action = random.choice(game.get_available_actions())  # TODO
            else:
                self.action = self.model.next_action(game)
            self.another_step, self.boxes = game.step(*self.action)
            self.steps += 1
            self.score_diff = game.get_player_score_diff()
            # Update epsilon
            if self.steps % self.epsilon_update_freq == 0:
                self.update_epsilon()
            return self.another_step
        else:
            return False
    
    def review_game(self, game):
        self.act(game)
        
    # Update epsilon (decrease exploration)
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # Train agent with replay memory
    def learn(self):
        if len(self.replay_memory) < self.min_replay_size:
            return
        
        minibatch = random.sample(self.replay_memory, self.batch_size)
        
        initial_states = np.array([transition[0] for transition in minibatch])
        initial_qs = self.model(torch.tensor(initial_states, dtype=torch.float32, device=self.device))

        next_states = np.array([transition[3] for transition in minibatch])
        target_qs = self.target_model(torch.tensor(next_states, dtype=torch.float32, device=self.device))

        states = torch.zeros((self.batch_size, self.model.state_size), dtype=torch.float32, device=self.device)
        updated_qs = torch.zeros((self.batch_size, self.model.action_size), dtype=torch.float32, device=self.device)

        for index, (state, action, reward, next_state, game_over) in enumerate(minibatch):
            if not game_over:
                max_future_q = reward + self.gamma * torch.max(target_qs[index])
            else:
                max_future_q = reward

            updated_qs_sample = initial_qs[index]
            updated_qs_sample[action] = max_future_q

            states[index] = torch.tensor(state, dtype=torch.float32, device=self.device)
            updated_qs[index] = updated_qs_sample

        predicted_qs = self.model(states)
        
        self.model.zero_grad()
        loss = self.loss_funct(predicted_qs, updated_qs)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
    # def learn(self):
    #     if len(self.replay_memory) < self.min_replay_size:
    #         return
    #     minibatch = random.sample(self.replay_memory, self.batch_size)
    #     # Optimize net
    #     for state, action, reward, next_state, game_over in minibatch:
    #         target = reward
    #         if not game_over:
    #             target += self.gamma * torch.max(self.target_model(torch.tensor(next_state, dtype=torch.float32, device=self.device)))  # TODO
    #         target_f = self.target_model(torch.tensor(state, dtype=torch.float32, device=self.device)).cpu().detach().numpy()
    #         target_f[action] = target
    #         self.model.zero_grad()
    #         loss = self.loss_funct(self.model(torch.tensor(state, dtype=torch.float32, device=self.device)), torch.tensor(target_f, dtype=torch.float32, device=self.device))
    #         loss.backward()
    #         self.optimizer.step()
    
"""
DQN Player
"""
class DQNPlayer(DQN):
    def reset(self):
        super().reset()
        
    def act(self, game):
        action = self.model.next_action(game)
        another_step, _ = game.step(*action)
        return another_step