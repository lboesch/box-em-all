"use client";

import confetti from "canvas-confetti";
import { useCallback, useEffect, useState } from "react";
import { initializeGame, makeMove } from "./actions";

const BOARD_SIZE = 3;

function randomInRange(min: number, max: number) {
  return Math.random() * (max - min) + min;
}

export default function DotsAndBoxes() {
  const [gameState, setGameState] = useState(null);
  const [showWinner, setShowWinner] = useState(false);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const startGame = async () => {
    const response = await initializeGame(BOARD_SIZE);
    const data = response;
    setGameState({
      board: data.initial_board,
      isGameOver: false,
      winner: null,
      currentPlayer: data.moves_made?.length > 0 ? 1 : data.current_player,
      scores: { 1: 0, 2: 0 },
    });
    if (data.moves_made?.length > 0) {
      //loop through moves made and update game state with time delay
      console.log("Moves made:", data.moves_made);
      data.moves_made.forEach((move, index) => {
        setTimeout(() => {
          console.log("Move:", move);
          setGameState({
            board: move.board_snapshot,
            isGameOver: false,
            currentPlayer: move.player_number,
            winner: null,
            scores: { 1: move.opponent_score, 2: move.your_score },
          });
        }, 1000 * index + 1);
      });
      if (
        data.moves_made[data.moves_made.length - 1].player_number !==
        data.current_player
      ) {
        const lastMove = data.moves_made[data.moves_made.length - 1];
        setTimeout(() => {
          setGameState({
            board: lastMove.board_snapshot,
            isGameOver: false,
            winner: null,
            scores: { 1: lastMove.opponent_score, 2: lastMove.your_score },
            currentPlayer: data.current_player,
          });
        }, 1000 * data.moves_made.length + 2);
      }
    }
    setSessionId(data.session_id);
    sessionStorage.setItem("sessionId", data.session_id);
  };

  const handleLineClick = useCallback(
    async (row: number, col: number) => {
      if (!gameState || gameState.isGameOver || gameState.currentPlayer !== 2)
        return;
      if (gameState.board[row][col] !== 0) return;
      console.log("Making move:", row, col);
      console.log("Session ID:", sessionId);
      const response = await makeMove(row, col, sessionId);

      const data = response;
      console.log("Move made:", data);
      if (data.moves_made?.length > 0) {
        console.log("Moves made:", data.moves_made);
        data.moves_made.forEach((move, index) => {
          setTimeout(() => {
            console.log("Move:", move);
            setGameState({
              board: move.board_snapshot,
              isGameOver: false,
              winner: null,
              currentPlayer: move.player_number,
              scores: { 1: move.opponent_score, 2: move.your_score },
            });
          }, 1000 * index + 1);
        });
        if (
          data.moves_made[data.moves_made.length - 1].player_number !==
          data.current_player
        ) {
          const lastMove = data.moves_made[data.moves_made.length - 1];
          setTimeout(() => {
            setGameState({
              board: lastMove.board_snapshot,
              isGameOver: false,
              winner: null,
              scores: { 1: lastMove.opponent_score, 2: lastMove.your_score },
              currentPlayer: data.current_player,
            });
          }, 1000 * data.moves_made.length + 2);
        }
      }
      if (data.message === "Game over") {
        setTimeout(() => {
          const lastMove = data.moves_made[data.moves_made.length - 1];
          const lastState = lastMove
            ? {
                board: lastMove.board_snapshot,
                isGameOver: false,
                winner: null,
                scores: { 1: lastMove.opponent_score, 2: lastMove.your_score },
              }
            : gameState;
          setGameState({
            ...lastState,
            isGameOver: true,
            winner: data.winner,
          });
        }, 1000 * (data.moves_made?.length || 0) + 8);
      }
    },
    [gameState, sessionId]
  );

  useEffect(() => {
    if (gameState?.isGameOver) {
      setShowWinner(true);
      let shape;
      let scalar = 1;
      const particleCount = 50;
      if (gameState.winner !== 2) {
        scalar = 3;
        shape = confetti.shapeFromText({ text: "😭", scalar });
      }
      const defaults = {
        origin: { y: 0.7 },
      };

      const isMobile = window.innerWidth <= 768;

      if (isMobile) {
        const interval = setInterval(() => {
          confetti(
            Object.assign({}, defaults, {
              shapes: shape ? [shape] : undefined,
              scalar: scalar,
              particleCount: 20,
              origin: { x: 0.5, y: 0.5 },
            })
          );
        }, 500);

        return () => clearInterval(interval);
      } else {
        const interval = setInterval(() => {
          confetti(
            Object.assign({}, defaults, {
              shapes: shape ? [shape] : undefined,
              scalar: scalar,
              particleCount,
              origin: { x: randomInRange(0.1, 0.3), y: Math.random() - 0.2 },
            })
          );
          confetti(
            Object.assign({}, defaults, {
              shapes: shape ? [shape] : undefined,
              scalar: scalar,
              particleCount,
              origin: { x: randomInRange(0.7, 0.9), y: Math.random() - 0.2 },
            })
          );
        }, 250);

        return () => clearInterval(interval);
      }
    }
  }, [gameState?.isGameOver, gameState?.winner]);

  useEffect(() => {
    if (!sessionId) {
      const storedSessionId = sessionStorage.getItem("sessionId");
      if (storedSessionId) {
        setSessionId(storedSessionId);
      }
    }
  }, [sessionId]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-4xl font-bold mb-8">Dots and Boxes</h1>
      {!gameState ? (
        <button
          className="px-4 py-2 bg-blue-500 text-white rounded-lg"
          onClick={startGame}
        >
          Start Game
        </button>
      ) : (
        <>
          <div className="flex flex-col items-center mb-4">
            <div className="relative flex items-center space-x-4">
              <div className="text-xl font-semibold">
                <span className="text-blue-500">PC Score:</span>{" "}
                {gameState.scores[1]}
              </div>
              <div className="text-xl font-semibold">
                <span className="text-red-500">Your Score:</span>{" "}
                {gameState.scores[2]}
              </div>
            </div>
            <div
              className={`transform transition-transform duration-300 ${
                gameState.currentPlayer === 1
                  ? "-translate-x-12"
                  : "translate-x-12"
              }`}
            >
              <span className="text-2xl">⬆️</span>
            </div>
          </div>
          <div className="bg-white p-8 rounded-lg shadow-lg">
            <div
              className="grid"
              style={{
                gridTemplateColumns: `repeat(${BOARD_SIZE * 2 + 1}, auto)`,
                gridTemplateRows: `repeat(${BOARD_SIZE * 2 + 1}, auto)`,
              }}
            >
              {gameState.board.map((row, rowIndex) =>
                row.map((cell, colIndex) => {
                  const isHorizontalLine =
                    rowIndex % 2 === 0 && colIndex % 2 !== 0;
                  const isVerticalLine =
                    rowIndex % 2 !== 0 && colIndex % 2 === 0;
                  const isBox = rowIndex % 2 !== 0 && colIndex % 2 !== 0;
                  if (isHorizontalLine) {
                    return (
                      <div
                        key={`${rowIndex}-${colIndex}`}
                        className={`w-16 h-2 cursor-pointer ${
                          cell === 0
                            ? "bg-gray-300 hover:bg-gray-400"
                            : cell === 1
                            ? "bg-blue-500"
                            : "bg-red-500"
                        }`}
                        onClick={() => handleLineClick(rowIndex, colIndex)}
                      />
                    );
                  }

                  if (isVerticalLine) {
                    return (
                      <div
                        key={`${rowIndex}-${colIndex}`}
                        className={`w-2 h-16 cursor-pointer ${
                          cell === 0
                            ? "bg-gray-300 hover:bg-gray-400"
                            : cell === 1
                            ? "bg-blue-500"
                            : "bg-red-500"
                        }`}
                        onClick={() => handleLineClick(rowIndex, colIndex)}
                      />
                    );
                  }

                  if (isBox) {
                    return (
                      <div
                        key={`${rowIndex}-${colIndex}`}
                        className={`w-16 h-16 flex items-center justify-center ${
                          cell === 1
                            ? "bg-blue-200"
                            : cell === 2
                            ? "bg-red-200"
                            : ""
                        }`}
                      />
                    );
                  }
                  return (
                    <div key={`${rowIndex}-${colIndex}`} className="w-2 h-2" />
                  );
                })
              )}
            </div>
          </div>
        </>
      )}
      {showWinner && (
        <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-50">
          <div className="bg-white p-8 rounded-lg shadow-lg text-center animate-fade-in">
            <h2 className="text-4xl font-bold text-green-500 mb-4 animate-bounce">
              {gameState.winner === 1
                ? "PC Wins!"
                : gameState.winner === 2
                ? "You have won!"
                : "It's a draw!"}
            </h2>
            <button
              className="mt-4 px-4 py-2 bg-blue-500 text-white rounded-lg"
              onClick={() => {
                setShowWinner(false);
                setGameState(null);
              }}
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
