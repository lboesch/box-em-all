from game import DotsAndBoxes
import model
import numpy as np
import player
import torch
from tqdm import tqdm
import wandb

# ====================================================================================================
# Global Parameters
# ====================================================================================================
debug = False
use_gpu = False
is_human = False
do_train = True
save_model = True
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
    board_size = 2
    episodes = 30000
    verification_episodes = 100
    ###
    alpha = 0.6  # TODO
    gamma = 0.5  # TODO
    epsilon = 1.0  # TODO
    epsilon_decay = 0.995  # TODO
    epsilon_min = 0.01  # TODO
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
        game_state="with-both-players",
        tags=["q-learning"]
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
        game = DotsAndBoxes(board_sizes=board_size, player_1=player_1, player_2=player_2)
        game.play(print_board=is_human)
        return
    
    # Game
    player_1 = player.GreedyPlayer('GreedyPlayer1')
    player_2 = player.QAgent('QAgent2', model=q_learning, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    game = DotsAndBoxes(board_size=board_size, player_1=player_1, player_2=player_2)
    verification_player_1 = player.GreedyPlayer('GreedyPlayer1')
    # verification_player_1 = player.RandomPlayer('RandomPlayer1')
    verification_player_2 = player.QPlayer('QPlayer2', model=q_learning)
    verification_game = DotsAndBoxes(board_size=board_size, player_1=verification_player_1, player_2=verification_player_2)
    
    rewards = []
    for episode in (pbar := tqdm(range(episodes if do_train else 1))):
        # Training
        pbar.set_description(f"Training Episode {episode}")
        if do_train:
            game.reset()
            game.play()
            rewards.append(game.player_2.total_reward)
    
        if debug or (episode > 0 and episode % 1000 == 0): 
            # Verification
            score = {'P1': 0, 'P2': 0, 'Tie': 0}
            for _ in range(verification_episodes):
                verification_game.reset()
                verification_game.play()
                
                # Update player score
                if verification_game.player_1.player_score > verification_game.player_2.player_score:
                    score['P1'] += 1
                elif verification_game.player_1.player_score < verification_game.player_2.player_score:
                    score['P2'] += 1
                else:
                    score['Tie'] += 1
        
            # Print verification score accross all verification episodes
            print("--------------------------------------------------------------------------------")
            print(f"Last Total Reward: {rewards[-1]}")
            print(f"Mean Total Rewards: {sum(rewards)/len(rewards)}")
            print("--------------------------------------------------------------------------------")
            print(f"Verification score in training episode {episode}: {score}")
            print(f"P1 ({verification_game.player_1.player_name}) Win: {round(score['P1'] / verification_episodes * 100, 2)}%")
            print(f"P2 ({verification_game.player_2.player_name}) Win: {round(score['P2'] / verification_episodes * 100, 2)}%")
            print("--------------------------------------------------------------------------------")
    
            # Export to Weigths & Biases
            if do_train and use_wandb:
                wandb.log(
                    step=episode,
                    data={
                        f"win-p1-{verification_game.player_1.player_name}": round(score['P1'] / verification_episodes * 100, 2),
                        f"win-p2-{verification_game.player_2.player_name}": round(score['P2'] / verification_episodes * 100, 2),
                        "tie": round(score['Tie'] / verification_episodes * 100, 2)
                    }
                )

    if do_train and save_model:
        # Save model
        q_learning.save(model_name_save)

# ====================================================================================================
# Deep Q-network (DQN)
# ====================================================================================================
# https://github.com/Floni/AI_dotsandboxes/blob/master/dotsandboxesplayer.py
# ====================================================================================================
def dqn():
    # Parameters
    board_size = 3
    episodes = 100000
    verification_episodes = 100
    ###
    alpha = 0.001  # TODO
    gamma = 0.5  # TODO
    epsilon = 1.0  # TODO
    epsilon_decay = 0.995  # TODO
    epsilon_min = 0.05  # TODO
    ###
    model_name_load = 'dqn_2_2'
    model_name_save = model.Policy.model_name('dqn_' + str(board_size))
    
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
        game_state="with-both-players",
        tags=["dqn"]
    )
    
    # Device
    if use_gpu:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():  # TODO very slow on Apple M1
            device = torch.device("mps")
    else:
        device = torch.device("cpu")
        
    # Model
    if do_train:
        # policy_net = model.DQN(board_size=board_size)
        policy_net = model.DQNConv(board_size=board_size)
    else:
        policy_net = model.DQN.load(model_name_load)
    policy_net.to(device)
    
    # Human Player
    if is_human:
        player_1 = player.HumanPlayer('HumanPlayer1')
        player_2 = player.DQNPlayer('DQNPlayer2', model=model.DQN.load("dqn_2_20241220085750"))
        game = DotsAndBoxes(board_size=board_size, player_1=player_1, player_2=player_2)
        game.play(print_board=is_human)
        return
    
    # Game
    player_1 = player.GreedyPlayer('GreedyPlayer1')
    # player_1 = player.QPlayer('QPlayer1', model=model.QLearning.load("q_learning_2_20241215001739"))
    # player_1 = player.DQNPlayer('DQNPlayer1', model=model.DQNConv.load("dqn_3_20250112162030"))
    player_2 = player.DQNAgent('DQNAgent2', model=policy_net, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min)
    game = DotsAndBoxes(board_size=board_size, player_1=player_1, player_2=player_2)
    verification_player_1 = player.GreedyPlayer('GreedyPlayer1')
    # verification_player_1 = player.RandomPlayer('RandomPlayer1')
    # verification_player_1 = player_1
    verification_player_2 = player.DQNPlayer('DQNPlayer2', model=policy_net)
    verification_game = DotsAndBoxes(board_size=board_size, player_1=verification_player_1, player_2=verification_player_2)
    
    rewards = []
    losses = []
    for episode in (pbar := tqdm(range(episodes if do_train else 1))):
        # Training
        pbar.set_description(f"Training Episode {episode}")
        policy_net.train()
        game.reset()
        game.play() 
        rewards.append(game.player_2.total_reward)
        losses.extend(game.player_2.losses)
        # TODO summary over all episode and reset_stats for training or verification
            
        if debug or (episode > 0 and episode % 1000 == 0): 
            # Verification
            score = {'P1': 0, 'P2': 0, 'Tie': 0}
            policy_net.eval()
            with torch.no_grad():
                for _ in range(verification_episodes):
                    verification_game.reset()
                    verification_game.play()
                    
                    # Update player score
                    if verification_game.player_1.player_score > verification_game.player_2.player_score:
                        score['P1'] += 1
                    elif verification_game.player_1.player_score < verification_game.player_2.player_score:
                        score['P2'] += 1
                    else:
                        score['Tie'] += 1
        
            # Print verification score accross all verification episodes
            print("--------------------------------------------------------------------------------")
            print(f"Last Loss: {losses[-1]}")
            print(f"Mean Last Losses: {sum(losses)/len(losses)}")
            print(f"Last Total Reward: {rewards[-1]}")
            print(f"Mean Total Rewards: {sum(rewards)/len(rewards)}")
            print("--------------------------------------------------------------------------------")
            print(f"Verification score in training episode {episode}: {score}")
            print(f"P1 ({verification_game.player_1.player_name}) Win: {round(score['P1'] / verification_episodes * 100, 2)}%")
            print(f"P2 ({verification_game.player_2.player_name}) Win: {round(score['P2'] / verification_episodes * 100, 2)}%")
            print("--------------------------------------------------------------------------------")
            
            # Export to Weigths & Biases
            if do_train and use_wandb:
                wandb.log(
                    step=episode,
                    data={
                        "train-loss:": losses[episode],
                        f"win-p1-{verification_game.player_1.player_name}": round(score['P1'] / verification_episodes * 100, 2),
                        f"win-p2-{verification_game.player_2.player_name}": round(score['P2'] / verification_episodes * 100, 2),
                        "tie": round(score['Tie'] / verification_episodes * 100, 2)
                    }
                )
                
    if do_train and save_model:
        # Save model
        policy_net.save(model_name_save)

# ====================================================================================================
# Start
# ====================================================================================================
if __name__ == "__main__":
    # q_learning()  # Q-learning
    dqn()  # Deep Q-network