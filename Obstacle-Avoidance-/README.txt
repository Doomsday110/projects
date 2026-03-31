This repository contains code for training and evaluating a reinforcement learning (RL) agent using the Proximal Policy Optimization (PPO) algorithm from Stable-Baselines3 to control a JetBot robot in a simulated environment powered by NVIDIA Isaac Sim. The environment is built using Gym and integrates with Isaac Sim for physics simulation and rendering.

Overview
The project consists of three main components:

Environment (env.py): A custom Gym environment (JetBotEnv) that simulates a JetBot robot in NVIDIA Isaac Sim. The environment handles physics, rendering, and interaction with the robot.
Training Script (train.py): Uses Stable-Baselines3's PPO algorithm to train a policy to control the JetBot. The policy is a CNN-based model that processes observations and outputs actions.
Evaluation Script (eval.py): Loads the trained policy and evaluates its performance in the simulated environment by running multiple episodes.
The goal is to train the JetBot to navigate or perform tasks in the simulation, with the policy learning from rewards and observations.

Prerequisites:
To run this code, you need the following:

Hardware: NVIDIA GPU with CUDA support.
Software:
NVIDIA Isaac Sim (installed and configured).
CUDA Toolkit (compatible with your GPU and Isaac Sim).
Python environment provided by NVIDIA Isaac Sim (this code is designed to run within Isaac Sim's Python interpreter).

Important Note:
This code is specifically tailored for the Python environment bundled with NVIDIA Isaac Sim. It relies on Isaac Sim's libraries (e.g., omni.isaac.kit, omni.isaac.core, omni.isaac.wheeled_robots) and will not work in a standard Python environment outside of Isaac Sim. Ensure you are running the scripts from within Isaac Sim's Python environment.

Installation:
1. Set up NVIDIA Isaac Sim:
2. Download and install NVIDIA Isaac Sim from the official website (https://developer.nvidia.com/isaac-sim).
3. Launch Isaac Sim and ensure it is fully configured, including the assets path and Omniverse connection.
4. Note the location of Isaac Sim's Python environment (typically found in the Isaac Sim installation directory,
   e.g., isaac-sim/python.sh or isaac-sim/python.bat).

Verify Dependencies:
The required dependencies (e.g., PyTorch, Stable-Baselines3, Gym) should already be included in Isaac Sim's Python environment. However, if you need to install additional packages, use the Python interpreter provided by Isaac Sim. For example:

On Linux/Mac:
/path/to/isaac-sim/python.sh -m pip install stable-baselines3 torch torchvision

On Windows:
"C:\path\to\isaac-sim\python.bat" -m pip install stable-baselines3 torch torchvision
Check Isaac Sim's documentation for the exact path to its Python environment.

Usage
Running Scripts in Isaac Sim
All scripts must be executed within the Isaac Sim Python environment. Do not use a standard Python installation or virtual environment, as Isaac Sim's libraries and configurations are required.

1. Training the Model
To train the PPO model, run the training script from within Isaac Sim:

Launch Isaac Sim.

Open a terminal or command prompt and navigate to the directory containing your scripts.

Use the Isaac Sim Python interpreter to run the script:
/path/to/isaac-sim/python.sh train.py

or on Windows:
"C:\path\to\isaac-sim\python.bat" train.py

The script will create a log directory (./cnn_policy) to store TensorBoard logs and model checkpoints.
Training parameters (e.g., learning rate, batch size) can be adjusted in train.py.
The final policy and checkpoints will be saved in the log directory.


2. Evaluating the Model
To evaluate the trained model, run the evaluation script similarly:
/path/to/isaac-sim/python.sh eval.py

or on Windows:
"C:\path\to\isaac-sim\python.bat" eval.py

This script loads the pre-trained model from ./cnn_policy/jetbot_policy.zip and runs 20 episodes in the simulation.
Set headless=False in eval.py to render the simulation visually. For non-headless mode, ensure your display is properly configured and Isaac Sim is launched with graphical support.


3. Monitoring Training
Use TensorBoard to monitor training progress. Launch TensorBoard from the same Isaac Sim Python environment:
/path/to/isaac-sim/python.sh -m tensorboard --logdir=./cnn_policy


Files
env.py: Defines the JetBotEnv class, a custom Gym environment for the JetBot in Isaac Sim.
train.py: Trains the PPO model using Stable-Baselines3.
eval.py: Evaluates the trained model in the simulation.
README.md: This file.


Configuration:
Environment Parameters: Adjust parameters like physics_dt, rendering_dt, and max_episode_length in env.py.
Policy Architecture: Modify the CNN layers and network architecture in train.py under policy_kwargs.
Training Hyperparameters: Tune PPO parameters (e.g., learning_rate, n_steps) in train.py.


Troubleshooting:
Isaac Sim Python Environment: Ensure you are using the Python interpreter bundled with Isaac Sim. Standard Python environments will lack the necessary Isaac Sim libraries.
CUDA Errors: Verify that your GPU, CUDA, and Isaac Sim versions are compatible. Update drivers or reinstall Isaac Sim if necessary.
Rendering Problems: If rendering fails in non-headless mode, ensure your display server is running and Isaac Sim is launched with graphical support. Check Isaac Sim's documentation for display configuration.
Module Not Found Errors: If you encounter missing module errors, ensure the Isaac Sim Python environment includes all required packages (e.g., stable-baselines3, torch). Install missing packages using the Isaac Sim Python interpreter.


License:
This project is licensed under the MIT License - see the  file for details.

Acknowledgments
NVIDIA Isaac Sim for providing the simulation platform.
Stable-Baselines3 for the PPO implementation.
PyTorch for deep learning support.


Contributions:
Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. For major changes, please open an issue first to discuss what you would like to change.