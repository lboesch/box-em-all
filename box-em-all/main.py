from game import DotsAndBoxes
import player
import model
import numpy as np
import os

def main():
    epochs = 100000
    score = {'P1': 0, 'P2': 0, 'Tie': 0}
    do_train = False
    
    # Learner / Model
    learner = model.QLearn(alpha = 0.1, gamma = 0.9, epsilon = 0.1)
    q_table = np.load('model/q_table_2_2.npy', allow_pickle=True).item()

    for _ in range(epochs):
        # Player
        player_1 = player.Human(1, 'Human1')
        # player_1 = player.ComputerRandom(1, 'Random1')
        # player_1 = player.ComputerGreedy(1, 'Greedy1')
        # player_2 = player.Human(2, 'Human2')
        # player_2 = player.ComputerGreedy(2, 'Greedy2')
        
        if do_train:
            player_2 = player.ComputerQLearner(2, 'QLearner2', learner)
        else:
            player_2 = player.ComputerQTable(2, 'QTable2', q_table)

        #Game
        game = DotsAndBoxes(rows=2, cols=2, player_1=player_1, player_2=player_2)
        game.play(print_board=True)
        
        if game.player_1.player_score > game.player_2.player_score:
            score['P1'] += 1
        elif game.player_1.player_score < game.player_2.player_score:
            score['P2'] += 1
        else:
            score['Tie'] += 1
    
    print("*******************")
    print("FinalScore: ", score)
    print("P1 Win: ", score['P1'] / epochs * 100)
    print("P2 Win: ", score['P2'] / epochs * 100)
    print("*******************")
    # print(learner.q_table)
    
    # print("blabla : " , os.getcwd())
    
    # Save Model
    if do_train:
        os.makedirs('model', exist_ok=True)
        filename = os.path.join('model', 'q_table_2_2')
        np.save(filename, learner.q_table)

if __name__ == "__main__":
    main()