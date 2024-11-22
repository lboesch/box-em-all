import numpy as np
from mctsnode import MCTSNode
import random
from game import DotsAndBoxes

class MCTS:
    def __init__(self, num_simulations=400):
        self.num_simulations = num_simulations

    def run_mcts(self, root_node):
        for _ in range(self.num_simulations):
            node = self.selection(root_node)

            if node.game_state.get_available_moves():
                self.expand(node)
                if node.children:  # If there are children, pick one for simulation
                    node = random.choice(node.children)
            reward = self.simulate(node)
            self.backpropagate(node, reward)

        # Choose the best child based on visit count
        return max(root_node.children, key=lambda n: n.visit_count)

    def selection(self, node):
        # Traverse the tree, selecting child nodes with the highest UCB1 score
        while node.children:
            node = max(node.children, key=lambda n: self.uct(n))
        return node

    def uct(self, node, exploration_param=1.41):
        # UCB1 formula for exploration vs. exploitation
        if node.visit_count == 0:
            return float('inf')  # Encourage exploration of unvisited nodes
        return (node.total_reward / node.visit_count) + exploration_param * \
               (np.sqrt(np.log(node.parent.visit_count) / node.visit_count))

    def expand(self, node):
        if not node.children:
            available_moves = node.game_state.get_available_moves()
            for move in available_moves:
                next_state = DotsAndBoxes(rows=node.game_state.rows, cols=node.game_state.cols, player_1=node.game_state.player_1, player_2=node.game_state.player_2, board=np.copy(node.game_state.board))
                next_state.make_move(*move)
                child_node = MCTSNode(next_state, parent=node, action=move)
                node.children.append(child_node)

    def simulate(self, node):
        # Simulate a random game starting from the given node
        game_state = node.game_state
        simualted_game_state = DotsAndBoxes(rows=game_state.rows, cols=game_state.cols, player_1=game_state.player_1, player_2=game_state.player_2, board=np.copy(game_state.board))

        while not simualted_game_state.is_game_over() and simualted_game_state.get_available_moves():
            move = random.choice(simualted_game_state.get_available_moves())
            simualted_game_state.make_move(*move)

        if simualted_game_state.scores[1] > simualted_game_state.scores[2]:
            return 1
        else:
            return -1

    def backpropagate(self, node, reward):
        # Propagate the result of the simulation up the tree
        while node:
            node.visit_count += 1
            node.total_reward += reward
            reward = -reward  # Alternate the reward to simulate the opponent’s perspective
            node = node.parent