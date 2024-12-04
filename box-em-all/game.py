import numpy as np
import random

# ====================================================================================================
# Dots & Boxes
# ====================================================================================================
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
        self.player_1.player_number = 1
        self.player_2 = player_2
        self.player_2.player_number = 2
        self.reset()
        
    def reset(self):
        # Reset board
        self.board = self.empty_board.copy()
        self.available_actions = self.all_actions.copy()
        # Reset players
        self.player_1.reset()
        self.player_2.reset()
        # self.current_player = self.player_2
        self.current_player = random.choice((self.player_1, self.player_2))
        self.opponent_player = self.player_2 if self.current_player == self.player_1 else self.player_1
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
    
    # Add edge
    def add_edge(self, row, col):
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
    def check_boxes(self, row, col, sim=None):
        boxes = {1: [], 2: [], 3: [], 4: []}
        if sim:  # TODO board as object to copy for simulation
           self.add_edge(row, col)  # Add edge (for simulation)
        if self.is_horizontal_edge(row, col):  # Horizontal edge
            for dx in [-1, 1]:
                row_offset = row + dx
                if 0 <= row_offset < self.board.shape[0]:
                    boxes[self.count_box_edges(row_offset, col)].append((row_offset, col))
        elif self.is_vertical_edge(row, col):  # Vertical edge
            for dy in [-1, 1]:
                col_offset = col + dy
                if 0 <= col_offset < self.board.shape[1]:
                    boxes[self.count_box_edges(row, col_offset)].append((row, col_offset))
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
    Actions
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
    
    # Return a shuffled list of available actions
    def get_random_available_actions(self):
        actions = self.get_available_actions()
        return random.sample(actions, len(actions))

    # Check if a action is valid
    def is_valid_action(self, row, col):
        return (row, col) in self.available_actions
    
    def get_action_by_idx(self, idx):
        return self.all_actions[idx]
        
    def get_idx_by_action(self, row, col):
        return self.all_actions.index((row, col))

    """
    Game
    """
    # Initialize board
    def __init_board(self):
        board = np.full(shape=(2 * self.rows + 1, 2 * self.cols + 1), fill_value=" ")
        board[::2, ::2] = "•"
        return board
    
    # Calculate game state size
    @staticmethod
    def calc_game_state_size(rows, cols):
        return rows * (cols + 1) + cols * (rows + 1)
    
    # Get current game state as flattened vector
    def get_game_state(self):
        return np.append(self.board[1::2, ::2] != ' ', self.board[::2, 1::2] != ' ').flatten().astype(int)
    
    # Get difference between player scores
    def get_player_score_diff(self):
        return self.current_player.player_score - self.opponent_player.player_score
    
    # Calculate reward
    def calc_reward(self, boxes, another_step, score_diff=None):
        reward = 0
        # Difference between player scores
        if score_diff is not None:
            reward += 0.1 * (self.get_player_score_diff() - score_diff)  
        # Box completed 
        # reward += 0.1 * len(boxes[4])
        # if another_step:
            # Chance to complete box with next action
            # reward += 0.1 * len(boxes[3])
        # else:
            # Giving advantage to opponent
            # reward -= 0.1 * len(boxes[3])
            # Drawing edge without completing a box
            # reward -= 0.1
        if self.is_game_over():  # TODO currently not possible to get a reward if the opponent has finished the game (MDP --> next state after opponents action?)
            # Winning a game
            if self.current_player.player_score > self.opponent_player.player_score:
                reward += 1
            # Loosing a game
            elif self.current_player.player_score < self.opponent_player.player_score:
                reward -= 1
        return reward
    
    # Perform a step
    def step(self, row, col):
        if self.is_valid_action(row, col):
            self.available_actions.remove((row, col))  # Remove action from list of available actions
            # Add edge
            self.add_edge(row, col)
            # Check for completed boxes
            boxes = self.check_boxes(row, col)
            if len(boxes[4]) > 0:
                for completed_box in boxes[4]:
                    self.board[completed_box] = str(self.current_player.player_number)
                    self.current_player.player_score += 1
                return True, boxes  # Box completed -> another step
            else:
                return False, boxes  # Box not completed -> switch player
        else:
            print("Invalid action. Try again.")
            return True, None  # Invalid action -> another step TODO create custom exception
    
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
            # Player action
            another_step = self.current_player.act(self)
            # Game over: Let the opponent review the game and then exit
            if self.is_game_over():
                self.switch_player()
                self.current_player.review_game(self)
                break
            # Switch player
            if not another_step:
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