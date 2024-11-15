import numpy as np
import random

class DotsAndBoxes:
    def __init__(self, rows, cols, player_1, player_2):
        # Initialize board
        self.rows = rows
        self.cols = cols
        self.total_boxes = self.rows * self.cols
        self.board = np.full(shape=(2 * self.rows + 1, 2 * self.cols + 1), fill_value=" ")
        self.board[::2, ::2] = "•"
        # Initialize available moves
        self.available_moves = self.get_available_moves()
        # Initialize players
        self.player_1 = player_1
        self.player_2 = player_2
        # self.current_player = self.player_1
        self.current_player = random.choice((self.player_1, self.player_2))
        # Initialize score
        # self.scores = {self.player_1.player_number: self.player_1.player_score, self.player_2.player_number: self.player_2.player_score}
                 
    # Prints the board     
    def print_board(self):
        print("\n")
        print("Current Board:")
        # TODO print(" ".join(map(str, list(range(0, len(self.board) + 2)))))
        for row in self.board:
            print(" ".join(row))
        print("\n")

        # print("Print 90g Board:")
        # rotated_board = np.rot90(self.board, k=90 // 90)
        # # TODO print(" ".join(map(str, list(range(0, len(self.board) + 2)))))
        # for row in rotated_board:
        #     print(" ".join(row))
        # print("\n")

        # print("Print 180g Board:")
        # rotated_board = np.rot90(self.board, k=180 // 90)
        # # TODO print(" ".join(map(str, list(range(0, len(self.board) + 2)))))
        # for row in rotated_board:
        #     print(" ".join(row))
        # print("\n")

        # state, index = self.get_canonical_state_and_rotation()
        # print("Print Canonical Board:")
        # print(state)
        # moves = self.get_canonical_moves(index);
        # print("canonical moves -> ", moves)

        # print("Print 180g Board:")
        # rotated_board = np.rot90(self.board, k=180 // 90)
        # # TODO print(" ".join(map(str, list(range(0, len(self.board) + 2)))))
        # for row in rotated_board:
        #     print(" ".join(row))
        # print("\n")

    def rotate_move(self, move, rotation_angle):
        row, col = move
        if rotation_angle == 90:
            return (col, self.board.shape[0] - 1 - row)
        elif rotation_angle == 180:
            return (self.board.shape[0] - 1 - row, self.board.shape[1] - 1 - col)
        elif rotation_angle == 270:
            return (self.board.shape[1] - 1 - col, row)
        else:
            return move  # No rotation

    # Get current game state as flattened vector
    def get_game_state(self):
        return np.append(self.board[1::2, ::2] != ' ', self.board[::2, 1::2] != ' ').flatten().astype(int)
    
    def get_game_state_with_board(self, board, index):
        return np.append(np.append(board[1::2, ::2] != ' ', board[::2, 1::2] != ' ').flatten().astype(int), index)

    def get_canonical_state_and_rotation(self):
        # Generate all rotated versions of the board
        rotations = [self.board]
        for i in range(1, 4):
            rotations.append(np.rot90(self.board, k=i))
        
        # Convert each rotation to bytes and find the smallest
        byte_representations = [rot.tobytes() for rot in rotations]
        min_index = byte_representations.index(min(byte_representations))
        
        # Return the canonical state and the rotation index that produces it
        canonical_bytes = byte_representations[min_index]
        return canonical_bytes, min_index
    
    def rotate_move(self, move, rotation):
        row, col = move
        n = self.board.shape[0]  # assuming a square board for simplicity
        if rotation == 1:  # 90 degrees
            return (col, n - 1 - row)
        elif rotation == 2:  # 180 degrees
            return (n - 1 - row, n - 1 - col)
        elif rotation == 3:  # 270 degrees
            return (n - 1 - col, row)
        return move  # 0 degrees (no rotation)
    
    def inverse_rotate_move(self, move, rotation):
        row, col = move
        n = self.board.shape[0]  # assuming a square board for simplicity
        if rotation == 1:  # 90 degrees inverse is 270 degrees
            return (n - 1 - col, row)
        elif rotation == 2:  # 180 degrees inverse is still 180 degrees
            return (n - 1 - row, n - 1 - col)
        elif rotation == 3:  # 270 degrees inverse is 90 degrees
            return (col, n - 1 - row)
        return move  # 0 degrees (no rotation)

    def get_canonical_moves(self, canonical_rotation):
        return [self.rotate_move(move, canonical_rotation) for move in self.available_moves]
        
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

    # Returns a list of available moves
    def get_available_moves(self):
        moves = []
        # Horizontal edges
        for row in range(0, self.board.shape[0], 2):
            for col in range(1, self.board.shape[1], 2):
                if self.is_edge_empty(row, col):
                    moves.append((row, col))
        # Vertical edges
        for row in range(1, self.board.shape[0], 2):
            for col in range(0, self.board.shape[1], 2):
                if self.is_edge_empty(row, col):
                    moves.append((row, col))
        return moves

    # Check if a move is valid
    def is_valid_move(self, row, col):
        return (row, col) in self.available_moves

    # Make a move
    def make_move(self, row, col):
        if self.is_valid_move(row, col):
            self.available_moves.remove((row, col))  # Remove move from available moves
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

    # Check if a box is completed
    # def check_boxes(self):
    #     completed_box = False
    #     for row in range(1, self.board.shape[0], 2):
    #         for col in range(1, self.board.shape[1], 2):
    #             if (
    #                 self.board[row - 1, col] == "-" and self.board[row + 1, col] == "-" and
    #                 self.board[row, col - 1] == "|" and self.board[row, col + 1] == "|"
    #             ):
    #                 if self.board[row, col] == " ":  # Only mark the box if it hasn't been claimed
    #                     self.board[row, col] = str(self.current_player.player_number)  # Mark the box with the player's number
    #                     self.current_player.player_score += 1
    #                     completed_box = True
    #     return completed_box

    # Switch player
    def switch_player(self):
        self.current_player = self.player_2 if self.current_player == self.player_1 else self.player_1

    # Check if game is over
    def is_game_over(self):
        return self.player_1.player_score + self.player_2.player_score == self.total_boxes

    # Play game
    def play(self, print_board=None):
        # Turn based game until game over
        # while not self.is_game_over():
        while self.available_moves:
            if print_board:
                self.print_board()
                print("--------------------------------------------------------------------------------")
                print(f"Available moves -> {self.available_moves}")
                print(f"Score -> Player 1: {self.player_1.player_score}, Player 2: {self.player_2.player_score}")
                print(f"Player -> {self.current_player.player_name}'s turn.")
                print("--------------------------------------------------------------------------------")
            another_move = self.current_player.play_turn(self)
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