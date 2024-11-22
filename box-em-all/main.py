from datetime import datetime
from game import DotsAndBoxes
import matplotlib.pyplot as plt
import model
import numpy as np
import player
from tqdm import tqdm
import wandb

def main():
    """
    Parameters
    """
    debug = True
    is_human = False
    do_train = True
    use_wandb = False
    extend_table = False

    board_size = 2
    epochs = 10000
    verification_epochs = 10

    alpha = 0.1
    gamma = 0.2
    epsilon = 0.3

    timestsamp = datetime.now().strftime("%Y%m%d%H%M%S")
    model_name_load = 'q_learning_3_20241117195021'
    model_name_save = 'q_learning_' + str(board_size) + '_' + timestsamp

    """
    Weights & Biases
    """
    if not is_human and do_train and use_wandb:
        wandb.login()
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
                "game-state": "monte carlo tree search",
            },
            tags=["q-learning", "dots-and-boxes"]
        )
    
    """
    Model
    """
    # if do_train:
    #     q_learning = model.QLearning(alpha=alpha, gamma=gamma, epsilon=epsilon, q_table={} if not extend_table else model.load(model_name_load).q_table)
    # else:
    #     q_learning = model.QLearning.load(model_name_load)
    
    """
    Human Player
    """    
    if is_human:
        player_1 = player.Human(1, 'Human1')
        player_2 = player.ComputerQTable(2, 'QTable2', q_learning)
        game = DotsAndBoxes(rows=board_size, cols=board_size, player_1=player_1, player_2=player_2)
        game.play(print_board=is_human)
        return
    
    """
    Training & Verification
    """
    # rewards = {}
    score = {'P1': 0, 'P2': 0, 'Tie': 0}
    for epoch in (pbar := tqdm(range(epochs if do_train else 1))):
        # Training
        pbar.set_description(f"Training Epoch {epoch}")
        if do_train:
            print("--------------------------------------------------------------------------------")
            #player_1 = player.ComputerGreedy(1, 'Greedy1')
            player_1 = player.ComputerRandom(1, 'Random1')
            player_2 = player.ComputerMCTS(2, 'MCTS')
            game = DotsAndBoxes(rows=board_size, cols=board_size, player_1=player_1, player_2=player_2)
            print("about to play")
            game.play()
            print("played")

            if game.scores[1] > game.scores[2]:
                score['P1'] += 1
            elif game.scores[1] < game.scores[2]:
                score['P2'] += 1
            else:
                score['Tie'] += 1

            print("--------------------------------------------------------------------------------")
            print(f"Final Score in epoch {epoch}: {score}")
            print(f"P1 ({game.player_1.player_name}) Win: {round(score['P1'] / (epoch+1) * 100, 2)}%")
            print(f"P2 ({game.player_2.player_name}) Win: {round(score['P2'] / (epoch+1) * 100, 2)}%")
            print("--------------------------------------------------------------------------------")       
    
        # Verification
        # for _ in range(verification_epochs):
            # verification_player_1 = player.ComputerGreedy(1, 'Greedy1')
            # verification_player_2 = player.ComputerMCTS(2, 'MCTS')
            # verification_game = DotsAndBoxes(rows=board_size, cols=board_size, player_1=verification_player_1, player_2=verification_player_2)
            # verification_game.play()
            # # Update player score
            # if verification_game.player_1.player_score > verification_game.player_2.player_score:
            #     score['P1'] += 1
            # elif verification_game.player_1.player_score < verification_game.player_2.player_score:
            #     score['P2'] += 1
            # else:
            #     score['Tie'] += 1
        
        # Verification results
        # if debug or ((epoch % 1000) == 0) and not use_wandb: 
        #     # Print final score accross all verification epochs
        #     print("--------------------------------------------------------------------------------")
        #     print(f"Final Score in epoch {epoch}: {score}")
        #     print(f"P1 ({verification_game.player_1.player_name}) Win: {round(score['P1'] / verification_epochs * 100, 2)}%")
        #     print(f"P2 ({verification_game.player_2.player_name}) Win: {round(score['P2'] / verification_epochs * 100, 2)}%")
        #     print("--------------------------------------------------------------------------------")          
    
        # Export to Weigths & Biases
        if not is_human and do_train and use_wandb:
            wandb.log({"epoch": epoch, "win-greedy": round(score['P1'] / verification_epochs * 100, 2), "win-qplayer": round(score['P2'] / verification_epochs * 100, 2), "tie": round(score['Tie'] / verification_epochs * 100, 2)})

    # if do_train:
        # Save model
        # q_learning.save(model_name_save)
        
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

    """
    Start
    """
if __name__ == "__main__":
    main()