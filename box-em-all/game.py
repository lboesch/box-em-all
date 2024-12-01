import numpy as np
import random

class DotsAndBoxes:
    def __init__(self, rows, cols, player_1, player_2):
        self.games_played = 0
        # Initialize board
        self.rows = rows
        self.cols = cols
        self.total_boxes = self.rows * self.cols
        self.empty_board = self.__init_board()
        self.all_actions = self.__init_actions()
        # Initialize players
        self.player_1 = player_1
        self.player_2 = player_2
        self.reset()
        
    def reset(self):
        # Reset board
        self.board = self.empty_board.copy()
        self.available_actions = self.all_actions.copy()
        # Reset players
        self.player_1.reset()
        self.player_2.reset()
        self.current_player = self.player_2
        self.opponent_player = self.player_2 if self.current_player == self.player_1 else self.player_1
        # self.current_player = random.choice((self.player_1, self.player_2))
        # Reset scoreboard
        # self.scores = {self.player_1.player_number: self.player_1.player_score, self.player_2.player_number: self.player_2.player_score}
    
    """
    Edges
    """    
    # Check if edge is empty
    def is_edge_empty(self, row, col):
        return self.board[row, col] == " "
    
    # Check if edge is horizontal
    def is_horizontal_edge(self, row, col):
        return row % 2 == 0 and col % 2 == 1
    
    # Check if edge is vertical
    def is_vertical_edge(self, row, col):
        return row % 2 == 1 and col % 2 == 0
    
    # Draw edge
    def draw_edge(self, row, col):
        if self.is_horizontal_edge(row, col):  # Horizontal edge
            self.board[row, col] = "-"
        elif self.is_vertical_edge(row, col):  # Vertical edge
            self.board[row, col] = "|"
    
    # Remove edge         
    def remove_edge(self, row, col):
        self.board[row, col] = " "
        
    """
    Boxes
    """
    # Check for any boxes that this edge might have completed
    def check_boxes(self, row, col, edges=4, sim=None):
        boxes = []
        if sim:
           self.draw_edge(row, col)  # Draw edge (for simulation)
        if self.is_horizontal_edge(row, col):  # Horizontal edge
            for dx in [-1, 1]:
                if 0 <= row + dx < self.board.shape[0] and self.count_box_edges(row + dx, col) == edges:
                    boxes.append((row + dx, col))
        elif self.is_vertical_edge(row, col):  # Vertical edge
            for dy in [-1, 1]:
                if 0 <= col + dy < self.board.shape[1] and self.count_box_edges(row, col + dy) == edges:
                    boxes.append((row, col + dy))
        if sim:
            self.remove_edge(row, col)  # Remove edge (for simulation)       
        return boxes
    
    def count_box_edges(self, x, y):
        edges = 0
        # Check if all edges of the box centered at (x, y) are filled
        if not self.is_edge_empty(x - 1, y):
            edges += 1
        if not self.is_edge_empty(x + 1, y):
            edges += 1
        if not self.is_edge_empty(x, y - 1):
            edges += 1
        if not self.is_edge_empty(x, y + 1):
            edges += 1   
        return edges

    """
    Actions / Moves
    """
    # Initialize actions
    def __init_actions(self):
        actions = []
        # Horizontal edges
        for row in range(0, self.empty_board.shape[0], 2):
            for col in range(1, self.empty_board.shape[1], 2):
                actions.append((row, col))
        # Vertical edges
        for row in range(1, self.empty_board.shape[0], 2):
            for col in range(0, self.empty_board.shape[1], 2):
                actions.append((row, col))
        return actions
    
    # Return a list of available actions
    def get_available_actions(self):
        return self.available_actions

    # Check if a action is valid
    def is_valid_action(self, row, col):
        return (row, col) in self.available_actions
    
    def get_action_by_idx(self, idx):
        return self.all_actions[idx]
        
    def get_idx_by_action(self, row, col):
        return self.all_actions.index((row, col))

    # Make a move
    # TODO rename to step
    def make_move(self, row, col):
        if self.is_valid_action(row, col):
            self.available_actions.remove((row, col))  # Remove action from available actions
            # Draw edge
            self.draw_edge(row, col)
            # Check for completed boxes
            completed_boxes = self.check_boxes(row, col)
            if len(completed_boxes) > 0:
                for completed_box in completed_boxes:
                    self.board[completed_box] = str(self.current_player.player_number)
                    self.current_player.player_score += 1
                return True  # Box completed -> another move
            else:
                return False  # Box not completed -> switch player
        else:
            print("Invalid move. Try again.")
            return True  # Invalid move -> another move

    """
    Game
    """
    # Initialize board
    def __init_board(self):
        board = np.full(shape=(2 * self.rows + 1, 2 * self.cols + 1), fill_value=" ")
        board[::2, ::2] = "•"
        return board
    
    @staticmethod
    def calc_game_state_size(rows, cols):
        return rows * (cols + 1) + cols * (rows + 1)
    
    # Get current game state as flattened vector
    def get_game_state(self):
        return np.append(self.board[1::2, ::2] != ' ', self.board[::2, 1::2] != ' ').flatten().astype(int)
    
    # Print the board
    def print_board(self):
        print("\n")
        print("Current Board:")
        # TODO print(" ".join(map(str, list(range(0, len(self.board) + 2)))))
        for row in self.board:
            print(" ".join(row))
        print("\n")
    
    # Switch player
    def switch_player(self):
        self.current_player, self.opponent_player = self.opponent_player, self.current_player

    # Check if game is over
    def is_game_over(self):
        return True if not self.get_available_actions() else False
        # return self.player_1.player_score + self.player_2.player_score == self.total_boxes

    # Play game
    def play(self, print_board=None):
        self.games_played += 1
        # Turn based game until game over
        while True:
            if print_board:
                self.print_board()
                print("--------------------------------------------------------------------------------")
                print(f"Available actions -> {self.get_available_actions()}")
                print(f"Score -> Player 1: {self.player_1.player_score}, Player 2: {self.player_2.player_score}")
                print(f"Player -> {self.current_player.player_name}'s turn.")
                print("--------------------------------------------------------------------------------")
            # Play turn
            another_move = self.current_player.play_turn(self)
            # Exit game and let the opponent player see the final action
            if self.is_game_over():
                self.switch_player()
                self.current_player.review_game(self)
                break
            # Switch player
            if not another_move:
                self.switch_player()

        # Game Over
        if print_board:
            self.print_board()
            print("--------------------------------------------------------------------------------")
            print("Game Over!")
            # Final Score
            print(f"Final Scores -> Player 1: {self.player_1.player_score}, Player 2: {self.player_2.player_score}")
            if self.player_1.player_score > self.player_2.player_score:
                print("Player 1 wins!")
            elif self.player_2.player_score > self.player_1.player_score:
                print("Player 2 wins!")
            else:
                print("It's a tie!")
            print("--------------------------------------------------------------------------------")