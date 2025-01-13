from flask import Flask, request, jsonify
from game import SinglePlayerOpponentDotsAndBoxes
import player
import uuid

app = Flask(__name__)

# Store game sessions
sessions = {}

@app.route('/start', methods=['POST'])
def start_game():
    data = request.json
    size = data.get('size')
    player_1 = player.GreedyPlayer('GreedyPlayer1')
    game = SinglePlayerOpponentDotsAndBoxes(board_size=size, opponent=player_1)
    initial_board = game.board.copy().tolist()
    game.play()
    session_id = str(uuid.uuid4())
    sessions[session_id] = game
    return jsonify({"message": "Game started", "session_id": session_id, "current_player": game.current_player, "moves_made": game.get_new_moves(), "initial_board": initial_board})

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

    if (game.current_player == game.opponent.player_number):
        game.play()

    if (row, col) not in game.get_available_actions():
        return jsonify({"message": "Invalid move"}), 400
    again = game.step(*(row, col));

    if not again:
        game.current_player = game.opponent.player_number
        game.play()
    if game.is_game_over():
        return jsonify({"message": "Game over", "moves_made": game.get_new_moves(), "winner": 1 if game.opponent_score > game.your_score else 2 if game.opponent_score < game.your_score else 0})
    return jsonify({"message": "Move made", "current_player": game.current_player, "moves_made": game.get_new_moves(), "board": game.board.tolist()})  


@app.route('/board', methods=['GET'])
def get_board():
    session_id = request.args.get('session_id')

    if session_id not in sessions:
        return jsonify({"message": "Invalid session ID"}), 400

    game = sessions[session_id]
    return jsonify({"board": game.board.tolist()})

if __name__ == '__main__':
    app.run(debug=True)