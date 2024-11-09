from game import DotsAndBoxes
import player

def main():
    # player_1 = player.Human(1, 'Human1')
    player_1 = player.ComputerGreedy(1, 'Greedy1')
    # player_2 = player.Human(2, 'Human2')
    player_2 = player.ComputerGreedy(2, 'Greedy2')
    game = DotsAndBoxes(rows=5, cols=5, player_1=player_1, player_2=player_2)
    game.play()

if __name__ == "__main__":
    main()