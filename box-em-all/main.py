from game import DotsAndBoxes
import model
import numpy as np
import player

def main():
    epochs = 100000
    score = {'P1': 0, 'P2': 0, 'Tie': 0}
    is_human = False
    do_train = False
    
    # Model
    if do_train:
        q_learning = model.QLearning(alpha = 0.1, gamma = -0.2, epsilon = 0.1)
    else:
        q_learning = model.load('q_table_2_2')

    for _ in range(epochs):
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

        # Play game
        game = DotsAndBoxes(rows=2, cols=2, player_1=player_1, player_2=player_2)
        game.play(print_board=is_human)
        
        # Update player score
        if game.player_1.player_score > game.player_2.player_score:
            score['P1'] += 1
        elif game.player_1.player_score < game.player_2.player_score:
            score['P2'] += 1
        else:
            score['Tie'] += 1
    
    # Print final score accross all epochs
    print("--------------------------------------------------------------------------------")
    print(f"Final Score: {score}")
    print(f"P1 ({game.player_1.player_name}) Win: {round(score['P1'] / epochs * 100, 2)}%")
    print(f"P2 ({game.player_2.player_name}) Win: {round(score['P2'] / epochs * 100, 2)}%")
    print("--------------------------------------------------------------------------------")
        
    # Save model
    if do_train:
        model.save('q_table_2_2', q_learning)

if __name__ == "__main__":
    main()