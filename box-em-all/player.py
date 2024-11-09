from abc import ABC, abstractmethod
import random
import numpy as np
import model

class Player(ABC):
    def __init__(self, player_number, player_name):
        self.player_number = player_number
        self.player_name = player_name
        self.player_score = 0
        
    @abstractmethod
    def play_turn(self, game):
        pass
    
# Human Player
class Human(Player): 
    def play_turn(self, game):
        # Let the human player choose a move using input
        try:
            row, col = map(int, input("Enter row and column for your move (e.g., 0 1): ").split())
            return game.make_move(row, col)
        except ValueError:
            print("Invalid input. Enter two integers separated by a space.")
            return True
        
# Random Player
class ComputerRandom(Player):
    def play_turn(self, game):
        move = ()
        # If no box-completing moves, pick a random available move
        move = random.choice(game.available_moves)   
        row, col = move    
        return game.make_move(row, col)
        
# Greedy Player
class ComputerGreedy(Player):
    def play_turn(self, game):
        move = ()
        # Prioritize moves that complete a box
        for available_move in game.available_moves:
            row, col = available_move
            if game.is_valid_move(row, col):
                completed_boxes = len(game.check_boxes(row, col, sim=True))
                if completed_boxes > 0:
                    move = (row, col)
                    break

        # If no box-completing moves, pick a random available move
        if not move:
            move = random.choice(game.available_moves)
            
        row, col = move    
    
        return game.make_move(row, col)

# Q Learning Player
class ComputerQLearner(Player):
    def __init__(self, player_number, player_name, learner):
        self.total_reward = 0
        self.learner = learner
        super().__init__(player_number, player_name)

    def play_turn(self, game):
        available_actions = game.get_available_moves()
        
        old_state = np.copy(game.board)            
                                  
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.learner.epsilon:
            action = random.choice(available_actions)
        else:
            action = max(available_actions, key=lambda x: self.learner.q_table.get((game.board.tobytes(), x), 0))
            
        row, col = action
        box_completed = game.make_move(row, col)
        
        boxes_4_edges = game.check_boxes(row, col, edges=4)
        boxes_3_edges = game.check_boxes(row, col, edges=3)

        # reward = 1 if box_completed else -1 if len(boxes_3_edges) > 0 else 0
        reward = 0
        if box_completed:
            reward += len(boxes_4_edges) + 0.5 * len(boxes_3_edges)
        else:
            reward -= 2 * len(boxes_3_edges)
        self.total_reward += reward

        self.learner.update_q_table(game, old_state, action, reward)

        return box_completed
    
# Q Table Player
class ComputerQTable(Player):
    def __init__(self, player_number, player_name, model):
        self.q_table = model
        super().__init__(player_number, player_name)

    def play_turn(self, game):
        available_actions = game.get_available_moves()
       
        action = max(available_actions, key=lambda x: self.q_table.get((game.board.tobytes(), x), 0))
            
        row, col = action    
        box_completed = game.make_move(row, col)

        return box_completed