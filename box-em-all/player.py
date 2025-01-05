from abc import ABC, abstractmethod
from collections import deque
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
        self.step_count = 0
        self.reset()

    def reset(self):
        self.player_score = 0  # TODO move to game?
        self.step = None
        self.step_hist = []
        
    @abstractmethod
    def act(self, game):
        pass
    
    def finalize_step(self, game):
        pass

"""
Agent Base Class
"""
# class Agent(Player):
#     def __init__(self, player_name, policy):
#         super().__init__(player_name)
#         self.policy = policy
    
#     def act(self, game):
#         action = self.policy.next_action(game)
#         another_step, _ = game.step(*action)
#         return another_step

"""
Learner Base Class
"""
# class Learner(Player):
#     def __init__(self, player_name, policy):
#         super().__init__(player_name)
#         self.policy = policy
        
#     @abstractmethod
#     def learn(self, game):
#         pass

# ====================================================================================================
# Players
# ====================================================================================================
"""
Human Player
"""
class HumanPlayer(Player):
    def act(self, game):
        # Let the human player choose an action using input
        try:
            action = map(int, input("Enter row and column for your action (e.g., 0 1): ").split())
            another_step = game.step(*action)
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
    
    def act(self, game):
        action = self.model.next_action(game)
        another_step = game.step(*action)
        return another_step
        
"""
Greedy Player
"""
class GreedyPlayer(Player):
    def __init__(self, player_name):
        super().__init__(player_name)
        self.model = model.Greedy()
    
    def act(self, game):
        action = action = self.model.next_action(game)
        another_step = game.step(*action)
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
        # Training
        self.epsilon_update_freq = 100
        
    def reset(self):
        super().reset()
        self.total_reward = 0

    def act(self, game):
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.epsilon:
            # Exploration
            self.action = random.choice(game.get_available_actions())
        else:
            # Exploitation
            self.action = self.model.next_action(game)
        # Step
        another_step = game.step(*self.action)
        return another_step
    
    def finalize_step(self, game):
        # Update total reward
        self.total_reward += self.step.reward
        # Update Q-table
        self.learn(game)
        # Update epsilon
        if self.step_count % self.epsilon_update_freq == 0:
            self.update_epsilon()
        
    # Update epsilon (decrease exploration)
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # Update Q-table (Q-learning algorithm)
    def learn(self, game):
        old_q_value = self.model.get_q_value(self.step.state, self.step.action)
        max_future_q = max([self.model.get_q_value(self.step.next_state, action) for action in game.get_available_actions()], default=0)
        # Formula: Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        new_q_value = old_q_value + self.alpha * (self.step.reward + self.gamma * max_future_q - old_q_value)  # TODO
        self.model.update_q_value(self.step.state, self.step.action, new_q_value)
    
"""
Q-learning Player
"""
class QPlayer(QLearning):
    def act(self, game):
        action = self.model.next_action(game)
        another_step = game.step(*action)
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
        # self.loss_funct = nn.MSELoss()
        self.loss_funct = nn.HuberLoss()
        # self.loss_funct = nn.SmoothL1Loss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=alpha)
        # self.optimizer = optim.Adam(self.model.parameters(), lr=alpha)
        # self.optimizer = optim.SGD(self.model.parameters(), lr=alpha)

        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon # Exploration rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        # Training
        self.batch_size = 128
        self.max_replay_size = 32 * self.batch_size
        self.min_replay_size = 8 * self.batch_size
        self.replay_memory = deque(maxlen=self.max_replay_size)
        self.model_update_freq = 16
        self.target_network_update_freq = 100
        self.epsilon_update_freq = 100
        self.last_loss = None
        
    def reset(self):
        super().reset()
        self.total_reward = 0
        
    def act(self, game):
        # Action
        if np.random.rand() <= self.epsilon:
            self.action = random.choice(game.get_available_actions())
        else:
            self.action = self.model.next_action(game)
        # Step
        self.another_step = game.step(*self.action)
        return self.another_step
    
    def finalize_step(self, game):
        # Update total reward
        self.total_reward += self.step.reward
        # Add step to replay memory
        self.replay_memory.append(self.step)
        # Learn
        if self.step_count % self.model_update_freq == 0:
            self.last_loss = self.learn()
        # Update target net
        if self.step_count % self.target_network_update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())
        # Update epsilon
        if self.step_count % self.epsilon_update_freq == 0:
            self.update_epsilon()

    # Update epsilon (decrease exploration)
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # Train agent with replay memory
    def learn(self):
        if len(self.replay_memory) < self.min_replay_size:
            return
        
        minibatch = random.sample(self.replay_memory, self.batch_size)
        
        initial_states = np.array([self.model.get_state(transition.state) for transition in minibatch])
        initial_qs = self.model(torch.tensor(initial_states, dtype=torch.float, device=self.device))

        next_states = np.array([self.target_model.get_state(transition.next_state) for transition in minibatch])
        target_qs = self.target_model(torch.tensor(next_states, dtype=torch.float, device=self.device))

        states = torch.zeros((self.batch_size, *self.model.input_shape), dtype=torch.float, device=self.device)
        updated_qs = torch.zeros((self.batch_size, self.model.action_size), dtype=torch.float, device=self.device)

        for index, step in enumerate(minibatch):
            if not step.game_over:
                max_future_q = step.reward + self.gamma * torch.max(target_qs[index])
            else:
                max_future_q = step.reward

            updated_qs_sample = initial_qs[index]
            updated_qs_sample[step.action] = max_future_q

            states[index] = torch.tensor(self.model.get_state(step.state), dtype=torch.float, device=self.device)
            updated_qs[index] = updated_qs_sample

        predicted_qs = self.model(states)
        
        self.model.zero_grad()
        loss = self.loss_funct(predicted_qs, updated_qs)
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
        
"""
DQN Player
"""
class DQNPlayer(DQN):
    def act(self, game):
        action = self.model.next_action(game)
        another_step = game.step(*action)
        return another_step