from abc import ABC, abstractmethod
import numpy as np
import random
from mcts import MCTS
from mctsnode import MCTSNode
from game import DotsAndBoxes

class Player(ABC):
    def __init__(self, player_number: int, player_name: str):
        self.player_number = player_number
        self.player_name = player_name
        self.player_score = 0  # TODO move to game
        
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

# Q-learning Player
class ComputerQLearning(Player):
    def __init__(self, player_number, player_name, model):
        super().__init__(player_number, player_name)
        self.model = model
        self.total_reward = 0

    def play_turn(self, game):
        old_state = np.copy(game.get_game_state())            
                                  
        # Epsilon-greedy action selection
        if random.uniform(0, 1) < self.model.epsilon:
            action = random.choice(game.available_moves)
        else:
            action = max(game.available_moves, key=lambda x: self.model.q_table.get((game.get_game_state().tobytes(), x), 0))          
        
        row, col = action
        
        another_move = game.make_move(row, col)
        count0, count1, boxes_2_edges, boxes_3_edges = game.count_boxes_with_n_edges()
        boxes_4_edges = game.check_boxes(row, col, edges=4)

        # Calculate reward
        # reward = 1 if another_move else -1 if len(boxes_3_edges) > 0 else 0
        reward =  0
        if another_move:
            reward += len(boxes_4_edges) + 0.5 * boxes_3_edges
        else:
            reward -= 4 * boxes_3_edges
        self.total_reward += reward

        # Update Q-table
        self.model.update_q_table(game, old_state, action, reward, another_move)

        return another_move
    
# Q-table Player
class ComputerQTable(Player):
    def __init__(self, player_number, player_name, model):
        super().__init__(player_number, player_name)
        self.model = model

    def play_turn(self, game):
        action = max(game.available_moves, key=lambda x: self.model.q_table.get((game.get_game_state().tobytes(), x), 0))
        row, col = action    
        another_move = game.make_move(row, col)
        return another_move

class ComputerMCTS(Player):
    def __init__(self, player_number, player_name, num_simulations=100):
        super().__init__(player_number, player_name)
        self.mcts = MCTS(num_simulations)

    def play_turn(self, game):
        # Create the root node with the current game state
        print(f"{self.player_name}: search move for board:")
        game.print_board()
        root_state = DotsAndBoxes(rows=game.rows, cols=game.cols, player_1=game.player_1, player_2=game.player_2, board=np.copy(game.board))
        root_node = MCTSNode(root_state)

        # Run MCTS to get the best action
        best_node = self.mcts.run_mcts(root_node)
        best_move = best_node.action

        # Make the selected move
        move = game.make_move(*best_move)

        print(f"{self.player_name} selected move: {best_move}")
        print(f"{self.player_name} board after move is:")
        game.print_board()
        return move
