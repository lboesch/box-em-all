from game import DotsAndBoxes
import matplotlib.pyplot as plt
import model
import numpy as np
import player
from tqdm import tqdm

def main():
    epochs = 100000
    score = {'P1': 0, 'P2': 0, 'Tie': 0}
    is_human = False
    do_train = True
    
    '''
    Initialization
    '''
    # Model
    if do_train:
        q_learning = model.QLearning(alpha = 0.2, gamma = -0.2, epsilon = 0.2)
    else:
        q_learning = model.QLearning.load('q_table_2_2')   
    # Player 1
    if is_human:
        player_1 = player.Human(1, 'Human1')
    else:
        if do_train:
            player_1 = player.ComputerRandom(1, 'Random1')
        else:
            player_1 = player.ComputerGreedy(1, 'Greedy1')     
    # Player 2
    if do_train:
        player_2 = player.ComputerQLearning(2, 'QLearning2', q_learning)
    else:
        player_2 = player.ComputerQTable(2, 'QTable2', q_learning) 
    # Game 
    game = DotsAndBoxes(rows=4, cols=4, player_1=player_1, player_2=player_2)

    '''
    Training & Evaluation
    '''
    rewards = {}
    for epoch in tqdm(range(epochs)):
        # Play game
        game.play(print_board=is_human)
        
        # Update player score
        if game.player_1.player_score > game.player_2.player_score:
            score['P1'] += 1
        elif game.player_1.player_score < game.player_2.player_score:
            score['P2'] += 1
        else:
            score['Tie'] += 1
         
        if do_train:   
            if epoch % (epochs / 100) == 0:
                # rewards[epoch] = list(rewards.values())[-1] + game.player_2.total_reward
                rewards[epoch] = game.player_2.total_reward
                    
        # Reset game
        game.reset()
        
    '''
    Results
    '''
    # Print final score accross all epochs
    print("--------------------------------------------------------------------------------")
    print(f"Final Score: {score}")
    print(f"P1 ({player_1.player_name}) Win: {round(score['P1'] / epochs * 100, 2)}%")
    print(f"P2 ({player_2.player_name}) Win: {round(score['P2'] / epochs * 100, 2)}%")
    print("--------------------------------------------------------------------------------")
    
    if do_train:
        # Save model
        q_learning.save('q_table_2_2')
        
        # Plot train stats
        x = list(rewards)
        y = list(rewards.values())
        z = np.polyfit(x, y, 1)
        p = np.poly1d(z)
        plt.plot(x, p(x))
        plt.plot(x, y)
        plt.xlabel('Epoch')
        plt.ylabel('Reward')
        plt.show()

    '''
    Start
    '''
if __name__ == "__main__":
    main()