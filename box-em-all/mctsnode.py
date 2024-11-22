class MCTSNode:
    def __init__(self, game_state, parent=None, action=None):
        self.game_state = game_state  # Store the state of the game at this node
        self.parent = parent  # Reference to the parent node
        self.action = action  # The action that led to this node
        self.children = []  # List of child nodes
        self.visit_count = 0  # Number of times this node was visited
        self.total_reward = 0  # Accumulated reward for this node