from dataclasses import dataclass
import numpy as np
import random

# ====================================================================================================
# Dots & Boxes
# ====================================================================================================
class DotsAndBoxes:
    def __init__(self, board_size, player_1, player_2):
        self.episode_count = 0
        # Initialize board
        self.board_size = board_size
        self.total_boxes = board_size ** 2
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
        self.current_player = self.player_2
        # self.current_player = random.choice((self.player_1, self.player_2))
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
        
    # Get horizontal edges
    @staticmethod
    def get_horizontal_edges(board):
        return (board[::2, 1::2] != ' ').astype(int)
    
    # Get vertical edges
    @staticmethod
    def get_vertical_edges(board):
        return (board[1::2, ::2] != ' ').astype(int)
        
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
    Steps
    """
    @dataclass
    class Step:
        state: np.ndarray = None
        state_score_diff: int = None
        action: tuple = None
        boxes: dict = None
        next_state: np.ndarray = None
        next_state_score_diff: int = None
        another_step: bool = None
        game_over: bool = None
        reward: int = None
        
        # Calculate reward
        def calc_reward(self, game):
            reward = 0
            # TODO http://www.papg.com/show?1TXA
            # Difference between player scores
            reward += 1 * (self.next_state_score_diff - self.state_score_diff)
            # Box completed
            # reward += 1 * len(self.boxes[4])
            if self.another_step:
                # Chance to complete box with next action
                reward += 0.5 * len(self.boxes[3])
            else:
                # Giving advantage to opponent
                reward -= 0.5 * len(self.boxes[3])
                # if len(self.boxes[3]) > 0:
                #     reward += 1 * (self.next_state_score_diff - self.state_score_diff)
            # Game over
            if self.game_over:
                # Winning a game
                if self.next_state_score_diff > 0:
                    reward += 1
                # Loosing a game
                elif self.next_state_score_diff < 0:
                    reward -= 1
            self.reward = reward
        
    # Perform a step
    def step(self, row, col):
        if self.is_valid_action(row, col):
            self.current_player.step_count += 1
            # Create new step
            step = self.current_player.step = self.Step()
            step.state = self.board.copy()  # TODO
            step.state_score_diff = self.get_player_score_diff()
            step.action = self.get_idx_by_action(row, col)  # TODO
            # Perform step
            self.available_actions.remove((row, col))  # Remove action from list of available actions
            self.add_edge(row, col)  # Add edge
            # Check for completed boxes
            step.boxes = self.check_boxes(row, col) 
            if len(step.boxes[4]) > 0:
                for completed_box in step.boxes[4]:
                    self.board[completed_box] = str(self.current_player.player_number)
                    self.current_player.player_score += 1
                another_step = True  # Box completed -> another step
            else:
                another_step = False  # Box not completed -> switch player
            step.another_step = another_step
        else:
            print("Invalid action. Try again.")
            another_step = True  # Invalid action -> another step TODO create custom exception
        return another_step
   
    # Finalize step
    def finalize_step(self):
        step = self.current_player.step
        if step:
            # Finalize step
            step.next_state = self.board.copy()  # TODO
            step.next_state_score_diff = self.get_player_score_diff()
            step.game_over = self.is_game_over()
            step.calc_reward(self)
            # Add step to history
            self.current_player.step_hist.append(step)
            # Let current player finalize the step
            self.current_player.finalize_step(self)
            # Reset step
            self.current_player.step = None
    
    """
    Players
    """
    # Get difference between player scores
    def get_player_score_diff(self):
        return self.current_player.player_score - self.opponent_player.player_score
    
    # Switch player
    def switch_player(self):
        self.current_player, self.opponent_player = self.opponent_player, self.current_player
        
    """
    Game
    """
    # Initialize board
    def __init_board(self):
        board = np.full(shape=(2 * self.board_size + 1, 2 * self.board_size + 1), fill_value=" ")
        board[::2, ::2] = "•"
        return board
    
    # Calculate game state size
    @staticmethod
    def calc_game_state_size(board_size):
        return 2 * board_size * (board_size + 1)
    
    # Get game state as 1d array
    @staticmethod
    def get_game_state(board):
        return np.append(DotsAndBoxes.get_horizontal_edges(board), DotsAndBoxes.get_vertical_edges(board)).flatten()
    
    # Get game state as 2d array (1 channel)
    @staticmethod
    def get_game_state_2d_1ch(board):
        return np.where((board == "-") | (board == "|"), 1, 0).astype(int)
        
    # Get game state as 2d array (2 channels)
    @staticmethod
    def get_game_state_2d_2ch(board):
        horizontal_edges = np.pad(DotsAndBoxes.get_horizontal_edges(board), ((0, 0), (0, 1)), mode="constant")
        vertical_edges = np.pad(DotsAndBoxes.get_vertical_edges(board), ((0, 1), (0, 0)), mode="constant")
        # horizontal_edges = np.where((board == "-"), 1, 0).astype(int)
        # vertical_edges = np.where((board == "|"), 1, 0).astype(int)
        return np.stack((horizontal_edges, vertical_edges))
    
    # Print the board
    def print_board(self):
        print("\n")
        print("Current Board:")
        # TODO print(" ".join(map(str, list(range(0, len(self.board) + 2)))))
        for row in self.board:
            print(" ".join(row))
        print("\n")

    # Check if game is over
    def is_game_over(self):
        return True if not self.get_available_actions() else False
        # return self.player_1.player_score + self.player_2.player_score == self.total_boxes

    # Play game
    def play(self, print_board=None):
        self.episode_count += 1
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
            # Switch player
            if not another_step:
                self.switch_player()
            # Finalize step
            self.finalize_step()
            # Game over
            if self.is_game_over():
                self.switch_player()
                self.finalize_step()
                break

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