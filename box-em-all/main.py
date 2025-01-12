from game import DotsAndBoxes, SinglePlayerOpponentDotsAndBoxes
import model
import numpy as np
import player
import torch
from tqdm import tqdm
import wandb
import os
import pickle
from collections import deque
import torch
from torch import nn
from torch.optim import Adam
import copy
from matplotlib import pyplot as plt
from matplotlib import animation
import random





# ====================================================================================================
# Global Parameters
# ====================================================================================================
debug = False
use_gpu = True
is_human = False
do_train = True
save_model = False
use_wandb = False
wandb_project = "box-em-all"
    
# ====================================================================================================
# Weight & Biases
# ====================================================================================================
def init_wandb(board_size, alpha, gamma, epsilon, epsilon_decay, epsilon_min, episodes, verification_episodes, model_name, game_state, tags):
    if not is_human and do_train and use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=wandb_project,
            # Track hyperparameters and run metadata
            config={
                "board-size": board_size,
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": epsilon,
                "epsilon-decay": epsilon_decay,
                "epsilon-min": epsilon_min,
                "episodes": episodes,
                "verification-episodes": verification_episodes,
                "model-name": model_name,
                "game-state": game_state,
            },
            tags=["dots-and-boxes"].append(tags)
        )

# ====================================================================================================
# Q-learning
# ====================================================================================================
def q_learning():
    # Parameters
    extend_q_table = False
    board_size = 3
    episodes = 100000000
    verification_episodes = 10
    ###
    alpha = 0.4  # TODO
    gamma = 0.9  # TODO
    epsilon = 0.95  # TODO
    epsilon_decay = 0.95  # TODO
    epsilon_min = 0.001  # TODO
    ###
    model_name_load = 'q_table_2_2'
    model_name_save = model.Policy.model_name('q_learning_' + str(board_size))

    # Weights & Biases
    init_wandb(
        board_size=board_size,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        episodes=episodes,
        verification_episodes=verification_episodes,
        model_name=model_name_save,
        game_state="try with complet new game algorithm",
        tags=["complete-refactoring"]
    )
    
    q_table={}  

    # Model
    if do_train:
        q_learning = model.QLearning(q_table={} if not extend_q_table else model.load(model_name_load).q_table)
    else:
        q_learning = model.QLearning.load(model_name_load)
    
    # Human Player
    if is_human:
        player_1 = player.HumanPlayer('HumanPlayer1')
        player_2 = player.QPlayer('QPlayer2', model=q_learning)
        game = DotsAndBoxes(board_sizes=board_size, player_1=player_1, player_2=player_2)
        game.play(print_board=is_human)
        return
    
    # Game
    
    player_1 = player.GreedyPlayer('GreedyPlayer1')
    #player_2 = player.QAgent('QAgent2', model=q_learning, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    game = SinglePlayerOpponentDotsAndBoxes(board_size=board_size, opponent=player_1)
    #game = DotsAndBoxes(board_size=board_size, player_1=player_1, player_2=player_2)
    #verification_player_1 = player.GreedyPlayer('GreedyPlayer1')
    # verification_player_1 = player.RandomPlayer('RandomPlayer1')
    #verification_player_2 = player.QPlayer('QPlayer2', model=q_learning)
    #verification_game = DotsAndBoxes(board_size=board_size, player_1=verification_player_1, player_2=verification_player_2)
    win_p1 = 0
    win_p2 = 0
    tie = 0
    
    for episode in (pbar := tqdm(range(episodes if do_train else 1))):
        # Training
        pbar.set_description(f"Training Episode {episode}")
        game.reset()
        game.play()
        steps = 0
        total_reward = 0
        while not game.is_game_over():
                # Keep track of the current step
            steps += 1

            # Store the previous observation
            game_state_before = game.get_game_state().copy()
            your_score_before = game.your_score
            opponent_score_before = game.opponent_score
            actions = game.get_available_actions().copy()
            np.random.shuffle(actions)

            if np.random.rand() > epsilon:
                # With probability 1-epsilon, choose randomly from the actions with
                # the highest Q
                action = max(actions, key=lambda x: q_table.get((game_state_before.tobytes(), x), 0))
            else:
                # With probability epsilon, choose a random action
                action = actions[np.random.choice(len(actions))]

            game.step(*action)
            game.play()

            game_state= game.get_game_state().copy()

            your_score = game.your_score
            opponent_score = game.opponent_score

            reward =(your_score - your_score_before) - (opponent_score - opponent_score_before)

            if (game.is_game_over()):
                if (game.your_score > game.opponent_score):
                    reward = 10
                elif (game.your_score < game.opponent_score):
                    reward = -10
                else:
                    reward += -1
            
            total_reward += reward

                # Q-learning formula
            next_actions = game.get_available_actions().copy()
            np.random.shuffle(next_actions)
            old_q_value = q_table.get((game_state_before.tobytes(), action), 0)
            max_future_q = max([q_table.get((game_state.tobytes(), a), 0) for a in next_actions], default=0)
            new_q_value = old_q_value + alpha * (reward + gamma * max_future_q - old_q_value)
            q_table[(game_state_before.tobytes(), action)] = new_q_value

            if episode % 100000 == 0:    
                if epsilon > epsilon_min:
                    epsilon *= epsilon_decay
    
        win = 1 if game.opponent_score > game.your_score else 2 if game.opponent_score < game.your_score else 0
        win_p1 += 1 if win == 1 else 0
        win_p2 += 1 if win == 2 else 0
        tie += 1 if win == 0 else 0
            # Export to Weigths & Biases
        if (episode % 100000 == 0):
            print("Episode {}, steps: {}, epsilon: {}, reward: {}".format(episode,
                                                         steps,
                                                         epsilon,
                                                         total_reward))

            if do_train and use_wandb:
                wandb.log(
                    step=episode,
                    data={
                        "epsilon": epsilon,
                        "total-reward:": total_reward,
                        "winner": win,
                        "win-p1": win_p1,
                        "win-p2": win_p2,
                        "tie": tie,
                        f"win-p1-rate": round(win_p1 / (100000) * 100, 2),
                        f"win-p2-rate": round(win_p2 / (100000) * 100, 2),
                        "tie-rate": round(tie / (100000) * 100, 2)
                        }

                    )
                win_p1 = 0
                win_p2 = 0
                tie = 0

    if do_train and save_model:
        # Save model
        base_path = 'model'
        os.makedirs(base_path, exist_ok=True)
        filename = os.path.join(base_path, 'q-table-learning-3x3.pkl')
        with open(filename, 'wb') as file:
            pickle.dump(q_table, file)
        # q_learning.save(model_name_save)

# ====================================================================================================
# Deep Q-network (DQN)
# ====================================================================================================
def linear_pairing(i, j, n):
    return i * n + j

def linear_unpairing(index, n):
    i = index // n
    j = index % n
    return i, j


def train(replay_memory, model, target_model, device, crit, optim, action_size):
    """Samples from the replay memory and updates the model
    """
    discount_factor = 0.99
    batch_size = 128
    max_replay_size = 32 * batch_size
    min_replay_size = 8 * batch_size
    model_update_freq = 4
    target_network_update_freq = 100
    train_episodes = 1000
    epsilon = 0.1
    learning_rate = 1e-4

    # Unless the replay memory is already at min_replay_size, do nothing
    if len(replay_memory) < min_replay_size:
        return

    # 11. Sample random minibatch of transitions from 𝑫
    mini_batch = random.sample(replay_memory, batch_size)

    # 12. Set 𝒚_𝒊= 𝒓_(𝒋+𝟏) 𝐢𝐟 𝐞𝐩𝐬𝐢𝐨𝐝𝐞 𝐭𝐞𝐫𝐦𝐢𝐧𝐚𝐭𝐞𝐬 𝐚𝐭 𝒔_(𝒋+𝟏)
    # Otherwise set 𝒚_𝒊 = 𝒓_(𝒋+𝟏)+𝜸𝐦𝐚𝐱_𝐚 𝐐(𝐬_(𝐣+𝟏), 𝐚;𝜽^−)

    # Get the Q values for the initial states of the trajectories from the model
    initial_states = np.array([transition[0] for transition in mini_batch])
    initial_qs = model(torch.tensor(initial_states, dtype=torch.float, device=device))

    # Get the "target" Q values for the next states
    next_states = np.array([transition[3] for transition in mini_batch])
    target_qs = target_model(torch.tensor(next_states, dtype=torch.float, device=device))

    # Prepare memory to fill in the states and q values after the update
    states = torch.zeros((batch_size, action_size), device=device)
    updated_qs = torch.zeros((batch_size, action_size), device=device)

    # For every transition in the mini batch
    for index, (observation,
                action,
                reward,
                new_observation,
                done) in enumerate(mini_batch):
        if not done:
            # If not terminal, include the next state
            max_future_q = reward + discount_factor * torch.max(
                target_qs[index])
        else:
            # If terminal, only include the immediate reward
            max_future_q = reward

        # The Qs for this sample of the mini batch
        updated_qs_sample = initial_qs[index]
        # Update the value for the taken action
        updated_qs_sample[action] = max_future_q

        # Keep track of the observation and updated Q value
        states[index] = torch.tensor(observation, dtype=torch.float, device=device)
        updated_qs[index] = updated_qs_sample

    # 13. Perform a gradient descent step on (𝒚_𝒊−𝑸(𝒔_𝒋, 𝒂_𝒋;𝜽))^𝟐 with respect
    # to network parameters 𝜽

    # For all of the mini batch
    predicted_qs = model(states)
    loss = crit(predicted_qs, updated_qs)

    optim.zero_grad()
    loss.backward()
    optim.step()

    # Return the loss from the mini batch for plotting
    return float(loss)


def dqn():
    # Parameters
    board_size = 2
    episodes = 100000
    verification_episodes = 10
    ###
    alpha = 0.1  # TODO
    gamma = 0.9  # TODO
    epsilon = 1.0  # TODO
    epsilon_decay = 0.995  # TODO
    epsilon_min = 0.05  # TODO
    learning_rate = 1e-4

    ###
    model_name_load = 'dqn_2_2'
    model_name_save = 'ueli-de-schwert'
    
    # Weights & Biases
    init_wandb(
        board_size=board_size,
        alpha=alpha,
        gamma=gamma,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min,
        episodes=episodes,
        verification_episodes=verification_episodes,
        model_name=model_name_save,
        game_state="complete new game algorithm",
        tags=["dqn"]
    )

    steps_since_model_update = 0
    steps_since_target_network_update = 0
    batch_size = 128
    max_replay_size = 32 * batch_size
    min_replay_size = 8 * batch_size
    model_update_freq = 4
    target_network_update_freq = 100

    # 2. Initialize replay memory
    replay_memory = deque(maxlen=max_replay_size)

    # 3. Initialize the Q network
    action_size = 2 * board_size * (board_size + 1)
    model = nn.Sequential(
            nn.Linear(action_size, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_size)
        )
    
    # Prepare a figure for plotting
    fig = plt.figure()
    ax = fig.add_subplot(211)
    ax.set_ylabel('Reward')

    ax2 = fig.add_subplot(212)
    ax2.set_xlabel('Episode No.')
    ax2.set_ylabel('Loss')


    # 4. Initialize the target network
    target_model = copy.deepcopy(model)

    # Prepare optimizer and loss function
    optim = Adam(model.parameters(), lr=learning_rate)
    crit = nn.MSELoss()
    
    # Device
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # TODO very slow on Apple M1
            device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    # Model
    model.to(device)
    target_model.to(device)
    
    # Game
    player_1 = player.GreedyPlayer('GreedyPlayer1')
    game = SinglePlayerOpponentDotsAndBoxes(board_size=board_size, opponent=player_1)
    win_p1 = 0
    win_p2 = 0
    tie = 0
    
    # Training
    rewards = []
    losses = []
    for episode in (pbar := tqdm(range(episodes if do_train else 1))):
        pbar.set_description(f"Training Episode {episode}")
        game.reset()
        game.play()

        steps = 0
        total_reward = 0
        total_training_rewards = 0
        steps_in_episode = 0

        while not game.is_game_over():
                # Keep track of the current step
            steps += 1

            # Store the previous observation
            game_state_before = game.get_game_state().copy()
            your_score_before = game.your_score
            opponent_score_before = game.opponent_score
            actions = game.get_available_actions().copy()
            np.random.shuffle(actions)
            if np.random.rand() > epsilon:
                # With probability 1-epsilon, choose randomly from the actions with
                # the highest Q
                state = torch.tensor(game_state_before, dtype=torch.float, device=device)
                q_values = model(state)
                action = max(game.get_available_actions(), key=lambda action: q_values[game.get_idx_by_action(*action)].item())
            else:
                # With probability epsilon, choose a random action
                action = actions[np.random.choice(len(actions))]

            actionIndex = game.get_idx_by_action(*action)

            game.step(*action)
            game.play()

            game_state= game.get_game_state().copy()

            your_score = game.your_score
            opponent_score = game.opponent_score

            reward =(your_score - your_score_before) - (opponent_score - opponent_score_before)

            if (game.is_game_over()):
                if (game.your_score > game.opponent_score):
                    reward = 10
                elif (game.your_score < game.opponent_score):
                    reward = -10
                else:
                    reward += -1
            
            total_reward += reward

             # Update no. steps taken
            steps_in_episode += 1
            steps_since_target_network_update += 1
            steps_since_model_update += 1

            replay_memory.append([game_state_before,
                              actionIndex,
                              reward,
                              game_state,
                              game.is_game_over()])
            
            if steps_since_model_update >= model_update_freq or game.is_game_over():
                last_loss = train(replay_memory, model, target_model, device, crit, optim, action_size)
                steps_since_model_update = 0

            total_training_rewards += reward

            if steps_since_target_network_update >= target_network_update_freq:
                target_model = copy.deepcopy(model)
                steps_since_target_network_update = 0

        win = 1 if game.opponent_score > game.your_score else 2 if game.opponent_score < game.your_score else 0
        win_p1 += 1 if win == 1 else 0
        win_p2 += 1 if win == 2 else 0
        tie += 1 if win == 0 else 0 

        # Add total reward for the episode and last training loss to list for plot
        rewards.append(total_training_rewards)
        losses.append(last_loss)

        if (episode > 0 and episode % 100 == 0):

            print("Episode {}, win-rate-p1: {}, epsilon: {}, reward: {}".format(episode,
                                                                                round(win_p1 / (100) * 100, 2),
                                            epsilon,
                                            np.max(rewards)))
            win_p1 = 0
            win_p2 = 0
            tie = 0
            if epsilon > epsilon_min:
                epsilon *= epsilon_decay
            # ax.plot(range(episode-100, episode), rewards[episode-99:], 'b.')
            # ax2.plot(range(episode-100, episode), losses[episode-99:], 'r.')
            # plt.show()

    
    # if do_train and save_model:
        # Save model
        # policy_net.save(model_name_save)

# ====================================================================================================
# Start
# ====================================================================================================
if __name__ == "__main__":
    #q_learning()  # Q-learning
    dqn()  # Deep Q-network