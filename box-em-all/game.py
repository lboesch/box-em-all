from dataclasses import dataclass
import time
from typing import Dict, List, Tuple
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
        self.all_actions = self.__init_actions(self.empty_board)
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
    
    """
    Edges
    """
    # Check if edge is empty
    @staticmethod
    def is_edge_empty(board, row, col):
        return board[row, col] == 0
    
    # Check if edge is horizontal
    @staticmethod
    def is_horizontal_edge(row, col):
        return row % 2 == 0 and col % 2 == 1
    
    # Check if edge is vertical
    @staticmethod
    def is_vertical_edge(row, col):
        return row % 2 == 1 and col % 2 == 0

    # Add edge
    def add_edge(self, board, row, col):
        board[row, col] = self.current_player.player_number

    # Remove edge
    @staticmethod
    def remove_edge(board, row, col):
        board[row, col] = 0
        
    # Get horizontal edges
    @staticmethod
    def get_horizontal_edges(board):
        return (board[::2, 1::2] > 0).astype(int)
    
    # Get vertical edges
    @staticmethod
    def get_vertical_edges(board):
        return (board[1::2, ::2] > 0).astype(int)
        
    """
    Boxes
    """
    # Check for any boxes that this edge might have completed
    def check_boxes(self, board, row, col):
        boxes = {1: [], 2: [], 3: [], 4: []}
        for box_row, box_col in self.get_adjacent_boxes(row, col):
            boxes[self.count_box_edges(board, box_row, box_col, row, col)].append((box_row, box_col))
        return boxes
    
    def get_box_edges(self, row: int, col: int) -> List[Tuple[int, int]]:
        edges = []
        if row > 0:
            edges.append((row - 1, col))  # top
        if row < self.board_size * 2:
            edges.append((row + 1, col))  # bottom
        if col > 0:
            edges.append((row, col - 1))  # left
        if col < self.board_size * 2:
            edges.append((row, col + 1))  # right
        return edges

    def get_adjacent_boxes(self, row: int, col: int) -> List[Tuple[int, int]]:
        adjacent_boxes = []
        if row % 2 == 0:  # Horizontal line
            if row > 0:
                adjacent_boxes.append((row - 1, col))
            if row < self.board_size * 2:
                adjacent_boxes.append((row + 1, col))
        else:  # Vertical line
            if col > 0:
                adjacent_boxes.append((row, col - 1))
            if col < self.board_size * 2:
                adjacent_boxes.append((row, col + 1))
        return adjacent_boxes

    def count_box_edges(self, board, row: int, col: int, row_check: int, col_check: int) -> int:
        return sum(1 for edge in self.get_box_edges(row, col) if board[edge] != 0 or edge == (row_check, col_check))

    """
    Actions
    """
    # Initialize actions
    @staticmethod
    def __init_actions(board):
        actions = []
        # Horizontal edges
        for row in range(0, board.shape[0], 2):
            for col in range(1, board.shape[1], 2):
                if DotsAndBoxes.is_edge_empty(board, row, col):
                    actions.append((row, col))
        # Vertical edges
        for row in range(1, board.shape[0], 2):
            for col in range(0, board.shape[1], 2):
                if DotsAndBoxes.is_edge_empty(board, row, col):
                    actions.append((row, col))
        return actions
    
    # Return a list of available actions
    def get_available_actions(self):
        return self.available_actions
    
    # Return a shuffled list of available actions
    def get_random_available_actions(self):
        actions = self.get_available_actions()
        return random.sample(actions, len(actions))
    
    # Get box completing actions
    def get_box_completing_actions(self, board, look_ahead) -> Dict[Tuple[int, int], int]:
        assert look_ahead > 0
        box_completing_actions = dict()
        for row, col in self.__init_actions(board):
            # Calculate the value of the action
            value = 0
            for box_row, box_col in self.get_adjacent_boxes(row, col):
                if self.count_box_edges(board, box_row, box_col, row, col) == 4:
                    value += 1

                    # Recursive call for look ahead
                    if look_ahead > 1:
                        # Simulate new action
                        new_board = board.copy()
                        self.add_edge(new_board, row, col)
                        next_actions = self.get_box_completing_actions(new_board, look_ahead=look_ahead - 1)
                        if next_actions:
                            value += max(next_actions.values())

            # Save action and value
            box_completing_actions[(row, col)] = value
                
        return box_completing_actions
    
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
                #     reward += 0.5 * len(game.get_box_completing_moves())
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
            step.state = self.board.copy()
            step.state_score_diff = self.get_player_score_diff()
            step.action = self.get_idx_by_action(row, col)
            # Perform step
            self.available_actions.remove((row, col))  # Remove action from list of available actions
            self.add_edge(self.board, row, col)  # Add edge
            # Check for completed boxes
            step.boxes = self.check_boxes(self.board, row, col) 
            if len(step.boxes[4]) > 0:
                for completed_box in step.boxes[4]:
                    self.board[completed_box] = self.current_player.player_number
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
            step.next_state = self.board.copy()
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
        board = np.zeros((self.board_size * 2 + 1, self.board_size * 2 + 1), dtype=np.int8)
        return board
    
    # Calculate game state size
    @staticmethod
    def calc_game_state_size(board_size):
        return 2 * board_size * (board_size + 1)
    
    # Get game state as 1d array
    @staticmethod
    def get_game_state(board):
        return np.append(DotsAndBoxes.get_horizontal_edges(board), DotsAndBoxes.get_vertical_edges(board)).flatten()
        
    # Get game state as 2d array (2 channels)
    @staticmethod
    def get_game_state_2d_2ch(board):
        horizontal_edges = np.pad(DotsAndBoxes.get_horizontal_edges(board), ((0, 0), (0, 1)), mode="constant")
        vertical_edges = np.pad(DotsAndBoxes.get_vertical_edges(board), ((0, 1), (0, 0)), mode="constant")
        return np.stack((horizontal_edges, vertical_edges))
    
    # Print the board
    def print_board(self):
        print("\n")
        print("Current Board:")
        for row in self.board:
            print(" ".join(map(str, row)))
        print("\n")
    
    def print_visual_board(self):
        """Prints the current game state with lines and boxes."""
        visual_board = []
        for row in range(self.board.shape[0]):
            visual_row = []
            for col in range(self.board.shape[1]):
                if row % 2 == 0 and col % 2 == 0:
                    visual_row.append("+")  # Dots
                elif row % 2 == 0:
                    visual_row.append("-" if self.board[row, col] != 0 else " ")  # Horizontal lines
                elif col % 2 == 0:
                    visual_row.append("|" if self.board[row, col] != 0 else " ")  # Vertical lines
                else:
                    visual_row.append(str(self.board[row, col]) if self.board[row, col] != 0 else " ")  # Empty space for boxes
            visual_board.append("".join(visual_row))
        print("\n".join(visual_board))

    # Check if game is over
    def is_game_over(self):
        return True if not self.get_available_actions() else False
    
    # Play game
    def play(self, print_board=None):
        self.episode_count += 1
        # Turn based game until game over
        while True:
            if print_board:
                self.print_visual_board()
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
            self.print_visual_board()
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

@dataclass
class Move:
    player_number: int
    row: int
    col: int
    timestamp: float
    board_snapshot: np.ndarray
    your_score: int
    opponent_score: int

class ApiPlayer:
    def __init__(self, player_name):
        self.player_name = player_name
        self.player_number = None
        self.player_score = 0

    def reset(self):
        self.player_score = 0
        self.step_count = 0

    def act(self, game):
        raise NotImplementedError("The act method should not be called on ApiPlayer.")

class SinglePlayerOpponentDotsAndBoxes(DotsAndBoxes):
    def __init__(self, board_size, opponent):
        dummy_player = ApiPlayer("DummyPlayer")
        super().__init__(board_size, opponent, dummy_player)
        self.reset()

    def reset(self):
        super().reset()
        self.move_history = []
        self.last_move_check = time.time()
    
    def play(self, _):
        raise NotImplementedError("Sorry my dear, but you have to implement game logic by yourself")

    def play_opponent(self): 
        if self.current_player.player_number == 1 and not self.is_game_over():
            return self.step_opponent()

    def step(self, row, col):
        if (row % 2 == 1 or col % 2 == 1) and self.board[row, col] == 0:  # Empty cell
            another_step = super().step(row, col)
            self.move_history.append(Move(
                player_number=self.current_player.player_number,
                row=row,
                col=col,
                timestamp=time.time(),
                board_snapshot=self.board.copy().tolist(),
                your_score=self.player_2.player_score,
                opponent_score=self.player_1.player_score
            ))
            if not another_step:
                self.switch_player()
            return another_step
        return False
    
    def step_opponent(self):
        self.current_player = self.player_1
        another_step = self.player_1.act(self) 
        if self.is_game_over():
            return
        if another_step:
            self.step_opponent()

    def get_moves(self) -> List[Move]:
        return self.move_history

    def get_new_moves(self):
        """Get moves that happened since last check"""
        new_moves = [move for move in self.move_history if move.timestamp > self.last_move_check]
        self.last_move_check = time.time()
        return new_moves
