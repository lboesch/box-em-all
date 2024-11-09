from game import DotsAndBoxes
import player
import model

def main():
    learner = model.QLearn(alpha = 0.1, gamma = 0.9, epsilon = 0.1)

    for _ in range(5):
        # player_1 = player.Human(1, 'Human1')
        player_1 = player.ComputerGreedy(1, 'Greedy1')
        # player_2 = player.Human(2, 'Human2')
        # player_2 = player.ComputerGreedy(2, 'Greedy2')
        player_2 = player.ComputerQLearner(2, 'QLearner2', learner)
        game = DotsAndBoxes(rows=2, cols=2, player_1=player_1, player_2=player_2)
        game.play()
        
    print(learner.q_table)

if __name__ == "__main__":
    main()