# Player Model

This document provides details about the trained player models, their training configurations, and performance metrics. Each model was trained using specific parameters and evaluated based on its win rate against a greedy player.

## Q-Learning Models

### Model: qlearning_2_greedy_61_greedy.pkl
- **Grid Size:** 2x2
- **Training Opponent:** Greedy
- **Training Episodes:** 100'000
- **Discount Factor (Gamma):** 0.5
- **Verification Win Rate:** 61% against a greedy player

## Deep Q-Network (DQN) Models

### Model: dqn_3_greedy_59_greedy.pt
- **Grid Size:** 3x3
- **Training Opponent:** Greedy
- **Training Episodes:** 100'000
- **Discount Factor (Gamma):** 0.5
- **Verification Win Rate:** 59% against a greedy player

## Deep Q-Network (with Convolutional Layers and Rainbow Extensions) Models

### Model: dqnconv_3_greedy_93_greedy.pt
- **Grid Size:** 3x3
- **Training Opponent:** Greedy
- **Training Episodes:** 100'000
- **Discount Factor (Gamma):** 0.5
- **Verification Win Rate:** 93% against a greedy player

### Model: dqnconv_3_greedy_94_greedy.pt
- **Grid Size:** 3x3
- **Training Opponent:** Greedy
- **Training Episodes:** 100'000
- **Discount Factor (Gamma):** 0.4
- **Verification Win Rate:** 94% against a greedy player

### Model: dqnconv_4_greedy_96_greedy.pt
- **Grid Size:** 4x4
- **Training Opponent:** Greedy
- **Training Episodes:** 100'000
- **Discount Factor (Gamma):** 0.5
- **Verification Win Rate:** 96% against a greedy player

### Model: dqnconv_5_greedy_95_greedy.pt
- **Grid Size:** 5x5
- **Training Opponent:** Greedy
- **Training Episodes:** 100'000
- **Discount Factor (Gamma):** 0.5
- **Win Rate:** 95% against a greedy player

### Model: dqnconv_5_greedy_ext_99_greedy_ext.pt
- **Grid Size:** 5x5
- **Training Opponent:** Greedy Extended (with look ahead)
- **Training Episodes:** 50'000
- **Discount Factor (Gamma):** 0.4
- **Win Rate:** 99% against a greedy (extended) player

---

### Notes
- **Win Rate** indicates the model's performance against a greedy player.
- **Gamma (Discount Factor)** is a key hyperparameter influencing the model's long-term reward optimization.