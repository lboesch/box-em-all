from game import DotsAndBoxes
import model
import numpy as np
import player
import statistics
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
use_wandb = True
wandb_project = "box-em-all"

# Device
if use_gpu:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():  # TODO very slow on Apple M1
        device = torch.device("mps")
else:
    device = torch.device("cpu")
    
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
    episodes = 100000
    verification_episodes = 100
    ###
    alpha = 0.6
    gamma = 0.4
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    ###
    model_name_load = 'qlearning_2_greedy_61_greedy'
    model_name_save = model.Policy.model_name('qlearning_' + str(board_size))

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
        game = DotsAndBoxes(board_size=board_size, player_1=player_1, player_2=player_2)
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
            print(f"Mean Total Rewards: {statistics.mean(rewards)}")
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
def dqn():
    # Parameters
    board_size = 3
    episodes = 100000
    verification_episodes = 100
    ###
    alpha = 0.001
    gamma = 0.4
    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.05
    ###
    model_name_load = 'dqnconv_3_greedy_94_greedy'
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
        
    # Model
    if do_train:
        # policy_net = model.DQN(board_size=board_size)
        policy_net = model.DQNConv(board_size=board_size)
    else:
        policy_net = model.DQNConv.load(model_name_load)
    policy_net.to(device)
    
    # Human Player
    if is_human:
        player_1 = player.HumanPlayer('HumanPlayer1')
        player_2 = player.GreedyPlayer('GreedyPlayer1')
        # player_2 = player.DQNPlayer('DQNPlayer2', model=model.DQNConv.load("dqnconf_3_greedy_94_greedy"))
        game = DotsAndBoxes(board_size=board_size, player_1=player_1, player_2=player_2)
        game.play(print_board=is_human)
        return
    
    # Game
    player_1 = player.GreedyPlayer('GreedyPlayer1')
    # player_1 = player.RandomPlayer('RandomPlayer1')
    # player_1 = player.QPlayer('QPlayer1', model=model.QLearning.load("qlearning_2_greedy_61_greedy"))
    # player_1 = player.DQNPlayer('DQNPlayer1', model=model.DQNConv.load("dqnconf_3_greedy_94_greedy"))
    player_2 = player.DQNAgent('DQNAgent2', model=policy_net, alpha=alpha, gamma=gamma, epsilon=epsilon, epsilon_decay=epsilon_decay, epsilon_min=epsilon_min, episodes=episodes)
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
            print(f"Mean Losses: {statistics.mean(losses)}")
            print(f"Last Total Reward: {rewards[-1]}")
            print(f"Mean Total Rewards: {statistics.mean(rewards)}")
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
                        "last-train-loss:": losses[-1],
                        "mean-train-loss:": statistics.mean(losses),
                        "last-train-reward:": rewards[-1],
                        "mean-train-reward:": statistics.mean(rewards),
                        f"win-p1-{verification_game.player_1.player_name}": round(score['P1'] / verification_episodes * 100, 2),
                        f"win-p2-{verification_game.player_2.player_name}": round(score['P2'] / verification_episodes * 100, 2),
                        "tie": round(score['Tie'] / verification_episodes * 100, 2)
                    }
                )
                
            if do_train and save_model:
                # Save model
                policy_net.save(model_name_save)
        
# ====================================================================================================
# Verification
# ====================================================================================================
def verification():
    # Parameters
    board_size = 3
    episodes = 10000
                        
    # Game
    # player_1 = player.RandomPlayer('RandomPlayer1')
    player_1 = player.GreedyPlayer('GreedyPlayer1')
    # player_1 = player.DQNPlayer('DQNPlayer1', model=model.DQNConv.load('dqnconv_3_greedy_94_greedy').eval().to(device))
    # player_2 = player.QPlayer('QPlayer1', model=model.QLearning.load("qlearning_2_greedy_61_greedy"))
    # player_2 = player.DQNPlayer('DQNPlayer2', model=model.DQN.load('dqn_3_greedy_59_greedy').eval().to(device))
    player_2 = player.DQNPlayer('DQNPlayer2', model=model.DQNConv.load('dqnconv_3_greedy_94_greedy').eval().to(device))
    game = DotsAndBoxes(board_size=board_size, player_1=player_1, player_2=player_2)
            
    score = {'P1': 0, 'P2': 0, 'Tie': 0}
    with torch.no_grad():
        for episode in (pbar := tqdm(range(episodes))):
            pbar.set_description(f"Verification Episode {episode}")
            game.reset()
            game.play()
            
            # Update player score
            if game.player_1.player_score > game.player_2.player_score:
                score['P1'] += 1
            elif game.player_1.player_score < game.player_2.player_score:
                score['P2'] += 1
            else:
                score['Tie'] += 1

    # Print verification score accross all episodes
    print("--------------------------------------------------------------------------------")
    print(f"Verification score after {episodes} episodes: {score}")
    print(f"P1 ({game.player_1.player_name}) Win: {round(score['P1'] / episodes * 100, 2)}%")
    print(f"P2 ({game.player_2.player_name}) Win: {round(score['P2'] / episodes * 100, 2)}%")
    print("--------------------------------------------------------------------------------")
            
# ====================================================================================================
# Start
# ====================================================================================================
if __name__ == "__main__":
    choice = input(
        "What do you want to do?\n" \
        "1) Q-learning\n" \
        "2) DQN\n" \
        "3) Verification\n" \
        "Enter your choice: "
    )
    
    if choice == "1":
        q_learning()  # Q-learning
    elif choice == "2":
        dqn()  # Deep Q-network
    elif choice == "3":
        verification()  # Verification