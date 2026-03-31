Q-Learning Agent for Grid World
This project implements a simple Q-Learning agent to navigate a small grid environment with rewards and penalties. The agent learns an optimal policy through reinforcement learning.

Project Description
The environment consists of a 2x3 grid with rewards:
Goal (100 points)
Hole (-100 points)
Other cells (-1 penalty per step)
The agent follows epsilon-greedy policy for exploration and exploitation.

The Q-table is updated using the Bellman equation.

Once trained, the agent follows the best policy to reach the goal.

Installation & Setup:
Install NumPy (if not already installed):
pip install numpy

Clone this repository:
git clone https://github.com/Doomsday110/Q_learning.git

Run the training script:
python Q_Learning_G21018083.py

Hyperparameters:
Parameter	Value
Alpha (α)	0.1
Gamma (γ)	0.9
Epsilon (ε)	0.1
Episodes	2000


The Q-Table
After training, the learned Q-table determines the best action for each state. The trained Q-table is displayed at the end of the script execution.

Running Policy After Training
Use the path_policy() function to determine the shortest path from any given start position.

Example:
path_policy(1, 0, Qtable, envr)

Agent navigates from (1,0) to Goal using optimal policy.

Future Improvements:
Implement larger grid environments.
Experiment with dynamic rewards.
Visualize the agent’s movement in real-time.

License:
This project is open-source. Feel free to use and modify it!
