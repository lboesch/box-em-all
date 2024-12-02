from datetime import datetime
from game import DotsAndBoxes
import matplotlib.pyplot as plt
import model
import numpy as np
import player
import torch
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
import wandb

# ====================================================================================================
# Global Parameters
# ====================================================================================================
debug = False
is_human = False
do_train = True
use_wandb = False
timestsamp = datetime.now().strftime("%Y%m%d%H%M%S")
if not is_human and do_train and use_wandb:
    wandb.login()

# ====================================================================================================
# Q-learning
# ====================================================================================================
def q_learning():
    # Parameters
    extend_q_table = False
    board_size = 3
    epochs = 100000
    verification_epochs = 100
    ###
    alpha = 0.1  # TODO
    gamma = 0.9  # TODO
    epsilon = 1.0  # TODO
    epsilon_decay = 0.995  # TODO
    epsilon_min = 0.1  # TODO
    ###
    model_name_load = 'q_table_2_2'
    model_name_save = 'q_learning_' + str(board_size) + '_' + timestsamp

    # Weights & Biases
    if not is_human and do_train and use_wandb:
        run = wandb.init(
            # Set the project where this run will be logged
            project="box-em-all",
            # Track hyperparameters and run metadata
            config={
                "board_size": board_size,
                "alpha": alpha,
                "gamma": gamma,
                "epsilon": epsilon,
                "epochs": epochs,
                "verification_epochs": verification_epochs,
                "game-state": "with-both-players",
            },
            tags=["q-learning", "dots-and-boxes"]
        )
    
    # Model
    if do_train:
        q_learning = model.QLearning(q_table={} if not extend_q_table else model.load(model_name_load).q_table)
    else:
        q_learning = model.QLearning.load(model_name_load)
    
    # Human Player
    if is_human:
        player_1 = player.HumanPlayer('HumanPlayer1')
        player_2 = player.QPlayer('QPlayer2', model=q_learning)
        game = DotsAndBoxes(rows=board_size, cols=board_size, player_1=player_1, player_2=player_2)
        game.play(print_board=is_human)
        return
    
    # Game
    player_1 = player.GreedyPlayer('GreedyPlayer1')
    player_2 = player.QAgent('QAgent2', model=q_learning, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    game = DotsAndBoxes(rows=board_size, cols=board_size, player_1=player_1, player_2=player_2)
    verification_player_1 = player.GreedyPlayer('GreedyPlayer1')
    verification_player_2 = player.QPlayer('QPlayer2', model=q_learning)
    verification_game = DotsAndBoxes(rows=board_size, cols=board_size, player_1=verification_player_1, player_2=verification_player_2)
    
    # rewards = {}
    for epoch in (pbar := tqdm(range(epochs if do_train else 1))):
        # Training
        pbar.set_description(f"Training Epoch {epoch}")
        if do_train:
            game.reset()
            game.play()
            if (epoch + 1) % 10 == 0:
                player_2.update_epsilon()
    
        # Verification
        if debug or ((epoch + 1) % 1000 == 0): 
            score = {'P1': 0, 'P2': 0, 'Tie': 0}
            for _ in range(verification_epochs):
                verification_game.reset()
                verification_game.play()
                
                # Update player score
                if verification_game.player_1.player_score > verification_game.player_2.player_score:
                    score['P1'] += 1
                elif verification_game.player_1.player_score < verification_game.player_2.player_score:
                    score['P2'] += 1
                else:
                    score['Tie'] += 1
        
            # Print verification score accross all verification epochs
            print("--------------------------------------------------------------------------------")
            print(f"Verification score in training epoch {epoch}: {score}")
            print(f"P1 ({verification_game.player_1.player_name}) Win: {round(score['P1'] / verification_epochs * 100, 2)}%")
            print(f"P2 ({verification_game.player_2.player_name}) Win: {round(score['P2'] / verification_epochs * 100, 2)}%")
            print("--------------------------------------------------------------------------------")
    
            # Export to Weigths & Biases
            if do_train and use_wandb:
                wandb.log({"epoch": epoch, "win-greedy": round(score['P1'] / verification_epochs * 100, 2), "win-qplayer": round(score['P2'] / verification_epochs * 100, 2), "tie": round(score['Tie'] / verification_epochs * 100, 2)})

    if do_train:
        # Save model
        q_learning.save(model_name_save)
        
        # Plot train stats
        # x = list(rewards)
        # y = list(rewards.values())
        # z = np.polyfit(x, y, 1)
        # p = np.poly1d(z)
        # plt.plot(x, p(x))
        # plt.plot(x, y)
        # plt.xlabel('Epoch')
        # plt.ylabel('Reward')
        # plt.show()

# ====================================================================================================
# Deep Q-network (DQN)
# ====================================================================================================
# https://pytorch.org/tutorials/beginner/saving_loading_models.html
# https://github.com/Floni/AI_dotsandboxes/blob/master/dotsandboxesplayer.py
# ====================================================================================================
def dqn():
    # Parameters
    board_size = 3
    epochs = 1000000
    verification_epochs = 100
    ###
    alpha = 0.1  # TODO
    gamma = 0.4  # TODO
    epsilon = 0.1  # TODO
    epsilon_decay = 0.995  # TODO
    epsilon_min = 0.1  # TODO
    ###
    model_name_load = 'dqn_2_2'
    model_name_save = 'dqn' + str(board_size) + '_' + timestsamp

    # Device TODO
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
     
    # Model
    state_size = DotsAndBoxes.calc_game_state_size(board_size, board_size)
    action_size = state_size
    policy_net = model.DQN(state_size, action_size)
    # TODO policy and target net
    # target_net = model.DQN(state_size, action_size)
    # target_net.load_state_dict(policy_net.state_dict())
    # target_net.eval()
        
    # optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    # replay_buffer = ReplayBuffer(MEMORY_CAPACITY)
    
    # Game
    player_1 = player.RandomPlayer('RandomPlayer1')
    player_2 = player.DQNAgent('DQNAgent2', model=policy_net, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    game = DotsAndBoxes(rows=board_size, cols=board_size, player_1=player_1, player_2=player_2)
    verification_player_1 = player.GreedyPlayer('GreedyPlayer1')
    verification_player_2 = player.DQNPlayer('DQNPlayer2', model=policy_net)
    verification_game = DotsAndBoxes(rows=board_size, cols=board_size, player_1=verification_player_1, player_2=verification_player_2)
    
    # Training
    for epoch in (pbar := tqdm(range(epochs if do_train else 1))):
        pbar.set_description(f"Training Epoch {epoch}")
        game.reset()
        game.play()
        if (epoch + 1) % 100 == 0:
            player_2.optimize(32)
            
        # Verification
        if debug or ((epoch + 1) % 1000 == 0): 
            score = {'P1': 0, 'P2': 0, 'Tie': 0}
            for _ in range(verification_epochs):
                verification_game.reset()
                verification_game.play()
                
                # Update player score
                if verification_game.player_1.player_score > verification_game.player_2.player_score:
                    score['P1'] += 1
                elif verification_game.player_1.player_score < verification_game.player_2.player_score:
                    score['P2'] += 1
                else:
                    score['Tie'] += 1
        
            # Print verification score accross all verification epochs
            print("--------------------------------------------------------------------------------")
            print(f"Verification score in training epoch {epoch}: {score}")
            print(f"P1 ({verification_game.player_1.player_name}) Win: {round(score['P1'] / verification_epochs * 100, 2)}%")
            print(f"P2 ({verification_game.player_2.player_name}) Win: {round(score['P2'] / verification_epochs * 100, 2)}%")
            print("--------------------------------------------------------------------------------")

# ====================================================================================================
# Start
# ====================================================================================================
if __name__ == "__main__":
    # q_learning()  # Q-learning
    dqn()  # Deep Q-network