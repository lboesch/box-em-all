from flask import Flask, request, jsonify
from game import SinglePlayerOpponentDotsAndBoxes
import player
import model
import uuid
import os
import pickle
from datetime import datetime

app = Flask(__name__)

sessions = {}  # fixme: use a redis or something

available_opponents = [
    {"name": "GreedyPlayer", "size": 2, "key": "greedy"},
    {"name": "RandomPlayer", "size": 2, "key": "random"},
    {"name": "GreedyPlayer", "size": 3, "key": "greedy"},
    {"name": "DQN", "size": 3, "key": "dqn3"},
    {"name": "GreedyPlayer", "size": 5, "key": "greedy"},
    {"name": "GreedyPlayer", "size": 7, "key": "greedy"}
]

opponent_classes = {
    "greedy": player.GreedyPlayer('GreedyPlayer'),
    "random": player.RandomPlayer('RandomPlayer'),
    "dqn3": player.DQNPlayer('DQNPlayer3', model=model.DQN.load('dqn_3', base_path='player-model'))
}


@app.route('/opponents', methods=['GET'])
def list_opponents():
    return jsonify({"available_opponents": available_opponents})

@app.route('/start', methods=['POST'])
def start_game():
    data = request.json
    size = data.get('size')
    opponent_key = data.get('opponent_key')

    opponent_info = next((opponent for opponent in available_opponents if opponent["size"] == size and opponent["key"] == opponent_key), None)

    if not opponent_info:
        return jsonify({"message": "Invalid opponent key or size. check /opponents for available options."}), 400

    opponent_class = opponent_classes.get(opponent_key)
    if not opponent_class:
        opponent_class = player.GreedyPlayer('GreedyPlayer1')
    
    game = SinglePlayerOpponentDotsAndBoxes(board_size=size, opponent=opponent_class)
    initial_board = game.board.copy().tolist()
    game.play_opponent()
    session_id = str(uuid.uuid4())
    sessions[session_id] = game
    return jsonify({"message": "Game started", "session_id": session_id, "current_player": game.current_player.player_number, "moves_made": game.get_new_moves(), "initial_board": initial_board})

@app.route('/move', methods=['POST'])
def make_move():
    data = request.json
    session_id = data.get('session_id')
    row = data.get('row')
    col = data.get('col')

    if session_id not in sessions:
        return jsonify({"message": "Invalid session ID"}), 400
    if not isinstance(row, int) or not isinstance(col, int):
        return jsonify({"message": "Row and column must be integers"}), 400

    game = sessions[session_id]

    if (game.current_player.player_number == game.player_1.player_number):
        game.play_opponent()

    if (row, col) not in game.get_available_actions():
        return jsonify({"message": "Invalid move"}), 400
    again = game.step(*(row, col));

    if not again:
        game.play_opponent()
    if game.is_game_over():
        game.get_moves()
        base_path = 'game-collection'
        os.makedirs(base_path, exist_ok=True)
        filename = os.path.join(base_path, game.player_1.player_name  + '_' + datetime.now().strftime("%Y%m%d%H%M%S") + '.pkl')
        with open(filename, 'wb') as file:
            pickle.dump(game.get_moves(), file)
        return jsonify({"message": "Game over", "moves_made": game.get_new_moves(), "winner": 1 if game.player_1.player_score > game.player_2.player_score else 2 if game.player_1.player_score < game.player_2.player_score else 0})
    return jsonify({"message": "Move made", "current_player": game.current_player.player_number, "moves_made": game.get_new_moves(), "board": game.board.tolist()})  


@app.route('/board', methods=['GET'])
def get_board():
    session_id = request.args.get('session_id')

    if session_id not in sessions:
        return jsonify({"message": "Invalid session ID"}), 400

    game = sessions[session_id]
    return jsonify({"board": game.board.tolist()})

if __name__ == '__main__':
    app.run(debug=True)