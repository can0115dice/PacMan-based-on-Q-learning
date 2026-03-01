# [PYTHON] Q-learning for playing Pac-Man

## Introduction
This project implements a Q-learning agent to play a simplified version of the Pac-Man game. The agent learns to navigate the game environment, collect dots, and avoid ghosts using the Q-learning algorithm.

## Project Structure
- `game_env.py`: Defines the Pac-Man game environment, including game logic, state representation, and reward mechanisms.
- `q_agent.py`: Implements the Q-learning agent, managing the Q-table, action selection (epsilon-greedy), and Q-value updates.
- `train.py`: Script to train the Q-learning agent and save the learned Q-table.
- `test .py`: Script to test a trained Q-learning agent with a Pygame visualization of the game.
- `model.pkl`: Stores the trained Q-table (pickle file).
- `training_result.png`: An image showing the results of the training process.

## Requirements
- Python 3.x
- NumPy
- Pygame
- Pickle (for model serialization)

## How to Use
### Training
To train the Q-learning model, run the `train.py` script. The script will train the agent and save the learned Q-table to `model.pkl`.
```bash
python train.py
```

### Testing
To test a trained model with visualization, run the `test .py` script. This script loads the `model.pkl` and visualizes the agent playing the game.
```bash
python test .py
```

### Controls during Testing
- **SPACE**: Pause / Resume the game.
- **ENTER / R**: Start a new game (after the current one ends).
- **↑ / ↓**: Adjust the AI's playing speed.
- **Q / ESC**: Quit the game.

## Game Design
- **Objective**: The Pac-Man agent aims to eat all dots on the map while avoiding ghosts. Eating all dots leads to victory.
- **Environment**: A 7x7 grid-based map without walls (open map).
- **Characters**: 
    - **Pac-Man**: The player-controlled agent.
    - **Ghost(s)**: One ghost that moves randomly.
- **Dots**: Randomly distributed on the map. The game ends when all dots are collected.
- **State Space**: The state is represented as a tuple containing:
    - Pac-Man's position.
    - Quadrant of Ghost 1.
    - Quadrant of Ghost 2 (if applicable, though the current implementation seems to have only one ghost).
    - Direction of the nearest dot.
    The total state space is approximately 8 (Pac-Man directions) * 3 (Ghost 1 distance levels) * 3 (Ghost 2 distance levels) * 4 (nearest dot directions) = 288 states. (Note: The `game_env.py` comments suggest 8 directions for Pac-Man, 3 distance levels for ghosts, and 4 directions for the nearest dot, leading to 8 * 3 * 3 * 4 = 288 states).
- **Action Space**: 4 actions: Up, Down, Left, Right.
- **Reward Function**: 
    - Eating a dot: Positive reward.
    - Being caught by a ghost: Negative reward (game over).
    - Moving: Small negative reward to encourage efficiency.

## Q-Learning Implementation
- **Algorithm**: Standard Q-learning algorithm.
- **Q-table**: Stores Q-values for each state-action pair. The Q-table is indexed by the state representation (pac position, ghost quadrants, nearest dot direction) and action.
- **Core Formula**: `Q(s,a) ← Q(s,a) + α * [r + γ * max_a' Q(s',a') - Q(s,a)]`
    - `α (alpha)`: Learning rate, controlling how much new information overrides old information.
    - `γ (gamma)`: Discount factor, determining the importance of future rewards.
- **Exploration**: Epsilon-greedy strategy.
    - `ε (epsilon)`: Probability of taking a random action for exploration.
    - `epsilon_decay`: Factor by which epsilon decays after each episode.
    - `epsilon_min`: Minimum value for epsilon to ensure continued exploration.
- **Parameters (from `q_agent.py` defaults)**:
    - `alpha`: 0.15
    - `gamma`: 0.95
    - `epsilon`: 1.0 (initial)
    - `epsilon_decay`: 0.995
    - `epsilon_min`: 0.05
- **Model Persistence**: The Q-table is saved and loaded using Python's `pickle` module.
