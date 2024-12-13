# Box 'Em All - An Approach to Solve Dots & Boxes with Reinforcement Learning

## Overview

Box 'Em All is a project that aims to solve the classic game of Dots & Boxes using reinforcement learning techniques. The game involves two players taking turns to draw lines between dots on a grid. The objective is to complete more boxes than the opponent. This project implements various strategies, including Q-learning, to train agents to play the game effectively.

## Game Rules

1. The game is played on a grid of dots.
2. Players take turns to draw a line between two adjacent dots, either horizontally or vertically.
3. When a player completes the fourth side of a box, they claim that box and earn a point.
4. The player who completes a box gets another turn.
5. The game ends when all boxes are completed.
6. The player with the most boxes at the end of the game wins.

## Project Structure


## Installation

1. **Clone the repository:**

    ```sh
    git clone https://github.com/yourusername/box-em-all.git
    cd box-em-all
    ```

2. **Install dependencies:**

    Make sure you have [Poetry](https://python-poetry.org/) installed. Then run:

    ```sh
    poetry install
    ```

3. **Activate the virtual environment:**

    ```sh
    poetry shell
    ```

## Running the Project

1. **Training the Model:**

    To train the Q-learning model, run:

    ```sh
    python box-em-all/main.py
    ```

    You can adjust the parameters in the [main](http://_vscodecontentref_/8) function of [main.py](http://_vscodecontentref_/9) to customize the training process.

2. **Playing the Game:**

    You can play the game against the trained model by setting [is_human](http://_vscodecontentref_/10) to `True` in the [main](http://_vscodecontentref_/11) function of [main.py](http://_vscodecontentref_/12) and then running:

    ```sh
    python box-em-all/main.py
    ```

3. **Using Weights & Biases:**

    If you want to log the training process using Weights & Biases, set [use_wandb](http://_vscodecontentref_/13) to `True` and make sure you are logged in to Weights & Biases:

    ```sh
    wandb login
    python box-em-all/main.py
    ```

## Customization

- **Board Size:** You can change the board size by modifying the [board_size](http://_vscodecontentref_/14) variable in the [main](http://_vscodecontentref_/15) function.
- **Hyperparameters:** Adjust the [alpha](http://_vscodecontentref_/16), [gamma](http://_vscodecontentref_/17), and [epsilon](http://_vscodecontentref_/18) values in the [main](http://_vscodecontentref_/19) function to tune the Q-learning algorithm.
- **Model Saving/Loading:** The model is saved and loaded from the [model](http://_vscodecontentref_/20) directory. You can change the model name by modifying the [model_name_load](http://_vscodecontentref_/21) and [model_name_save](http://_vscodecontentref_/22) variables.
