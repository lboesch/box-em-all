from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, List, Tuple
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
        self.epsilon_max = epsilon
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
            action = random.choice(game.get_available_actions())
        else:
            # Exploitation
            action = self.model.next_action(game)
        # Step
        another_step = game.step(*action)
        return another_step
    
    def finalize_step(self, game):
        # Update total reward
        self.total_reward += self.step.reward
        # Update Q-table
        self.learn(game)
        
    # Update epsilon (decrease exploration)
    def update_epsilon(self):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    # Update Q-table (Q-learning algorithm)
    def learn(self, game):
        old_q_value = self.model.get_q_value(self.step.state, self.step.action)
        max_future_q = max([self.model.get_q_value(self.step.next_state, game.get_idx_by_action(*action)) for action in game.get_available_actions()], default=0)
        # Formula: Q(s,a) = Q(s,a) + α * (r + γ * max(Q(s',a')) - Q(s,a))
        new_q_value = old_q_value + self.alpha * (self.step.reward + self.gamma * max_future_q - old_q_value)
        self.model.update_q_value(self.step.state, self.step.action, new_q_value)
        
        # Update epsilon
        if self.step_count % self.epsilon_update_freq == 0:
            self.update_epsilon()
    
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
        self.epsilon_max = epsilon
        self.epsilon_min = epsilon_min
        # Training
        self.n_step = 3
        self.batch_size = 64
        self.max_replay_size = 32 * self.batch_size
        self.min_replay_size = 8 * self.batch_size
        self.model_update_freq = 16
        self.target_network_update_freq = 100
        self.epsilon_update_freq = 100
        # memory for 1-step Learning
        self.memory = ReplayBuffer(
            self.model.input_shape, self.max_replay_size, self.batch_size, n_step=1, gamma=gamma
        )
        # memory for N-step Learning
        self.use_n_step = True if self.n_step > 1 else False
        if self.use_n_step:
            self.memory_n = ReplayBuffer(
                self.model.input_shape, self.max_replay_size, self.batch_size, n_step=self.n_step, gamma=gamma
            ) 
        self.transition = list()
        
    def reset(self):
        super().reset()
        self.total_reward = 0
        self.losses = [] 
        
    def act(self, game):
        # Action
        if np.random.rand() <= self.epsilon:
            action = random.choice(game.get_available_actions())
        else:
            action = self.model.next_action(game)
        # Step
        another_step = game.step(*action)
        return another_step
    
    def finalize_step(self, game):
        # Update total reward
        self.total_reward += self.step.reward
        # Add step to replay memory
        self.transition = [
            self.model.get_state(self.step.state),
            self.step.action,
            self.step.reward,
            self.model.get_state(self.step.next_state),
            self.step.game_over
        ]     
        # N-step transition
        if self.use_n_step:
            one_step_transition = self.memory_n.store(*self.transition)
        # 1-step transition
        else:
            one_step_transition = self.transition
        # add a single step transition
        if one_step_transition:
            self.memory.store(*one_step_transition)
        # Learn
        if self.step_count % self.model_update_freq == 0:
            self.learn()

    # Update epsilon (decrease exploration)
    def update_epsilon(self):
        # linearly decrease epsilon
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon - (self.epsilon_max - self.epsilon_min) * self.epsilon_decay
        )
    
    def learn(self):
         # if training is ready
        if len(self.memory) >= self.min_replay_size:  # TODO self.batch_size?
            loss = self.update_model()
            self.losses.append(loss)
            # update_cnt += 1  # TODO instead of step_count?
            
            # Update epsilon
            if self.step_count % self.epsilon_update_freq == 0:
                self.update_epsilon()
            
            # Update target net
            if self.step_count % self.target_network_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

    # Train agent with replay memory
    def update_model(self):        
        samples = self.memory.sample_batch()
        indices = samples["indices"]
        
        # 1-step Learning loss
        loss = self._compute_dqn_loss(samples, self.gamma)
          
        # N-step Learning loss
        if self.use_n_step:
            samples = self.memory_n.sample_batch_from_idxs(indices)
            gamma = self.gamma ** self.n_step
            n_loss = self._compute_dqn_loss(samples, gamma)
            loss += n_loss  # combining 1-step loss and n-step loss to prevent high-variance
            
        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        return loss.item()
        
    def _compute_dqn_loss(self, samples, gamma):
        state = torch.FloatTensor(samples["obs"]).to(self.device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(self.device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(self.device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(self.device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(self.device)
        
        curr_q_value = self.model(state).gather(dim=1, index=action)
        next_q_value = self.target_model(next_state).gather(  # Double DQN
            dim=1, index=self.model(next_state).argmax(dim=1, keepdim=True)
        ).detach()
        target = reward + gamma * next_q_value * (1 - done)
        
        loss = self.loss_funct(curr_q_value, target)
        
        return loss
        
"""
DQN Player
"""
class DQNPlayer(DQN):
    def act(self, game):
        action = self.model.next_action(game)
        another_step = game.step(*action)
        return another_step
    
# ====================================================================================================
# Replay buffer
# ====================================================================================================
class ReplayBuffer:
    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        n_step: int = 3, 
        gamma: float = 0.99,
    ):
        self.obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, *obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0
        
        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool
    ) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray, bool]:
        transition = (obs, act, rew, next_obs, done)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()
        
        # make a n-step transition
        rew, next_obs, done = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]
        
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
        
        return self.n_step_buffer[0]

    def sample_batch(self) -> Dict[str, np.ndarray]:
        indices = np.random.choice(
            self.size, size=self.batch_size, replace=False
        )

        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
            # for N-step Learning
            indices=indices,
        )
    
    def sample_batch_from_idxs(
        self, indices: np.ndarray
    ) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[indices],
            next_obs=self.next_obs_buf[indices],
            acts=self.acts_buf[indices],
            rews=self.rews_buf[indices],
            done=self.done_buf[indices],
        )
    
    def _get_n_step_info(
        self, n_step_buffer: Deque, gamma: float
    ) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done"""
        # info of the last transition
        rew, next_obs, done = n_step_buffer[-1][-3:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o, d = transition[-3:]

            rew = r + gamma * rew * (1 - d)
            next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs, done

    def __len__(self) -> int:
        return self.size