from abc import ABC, abstractmethod
import random

class Player(ABC):
    def __init__(self, player_number, player_name):
        self.player_number = player_number
        self.player_name = player_name
        self.player_score = 0
        
    @abstractmethod
    def choose_move(self, game):
        pass
    
# Human Player
class Human(Player): 
    def choose_move(self, game):
        # Let the human player choose a move using input
        return map(int, input("Enter row and column for your move (e.g., 0 1): ").split())
        
# Greedy Player
class ComputerGreedy(Player):
    def choose_move(self, game):
        # Prioritize moves that complete a box
        for move in game.available_moves:
            row, col = move
            if game.is_valid_move(row, col):
                game.draw_edge(row, col)  # TODO remove workaround
                completed_boxes = len(game.check_for_completed_boxes(row, col))
                game.remove_edge(row, col)  # TODO remove workaround
                if completed_boxes > 0:
                    return (row, col)
                
            # if game.add_edge(row, col, self.player_num):
            #     return (row, col)

        # If no box-completing moves, pick a random available move
        return random.choice(game.available_moves)

# AI Player
class ComputerAi(Player):  
    def choose_move(self, game):
       pass