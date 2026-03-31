from stable_baselines3 import PPO
from stable_baselines3.ppo import CnnPolicy
from stable_baselines3.common.callbacks import CheckpointCallback
import torch as th
from environment import HelloWorld

# Log directory of the tensorboard files to visualize the training and for the final policy as well
log_dir = "./cnn_policy"

# Assuming my_env and log_dir are properly defined within the HelloWorld class
hello_world_instance = HelloWorld()

# Using the HelloWorld instance to access the JetBot environment
my_env = hello_world_instance.jetbot_env

# Define the CNN policy architecture
policy_kwargs = dict(
    features_extractor_kwargs=dict(
        cnn_layers=[
            # Define your CNN layers here
            {"filters": 32, "kernel_size": 8, "strides": 4, "activation": th.nn.Tanh},
            {"filters": 64, "kernel_size": 4, "strides": 2, "activation": th.nn.Tanh},
            {"filters": 64, "kernel_size": 3, "strides": 1, "activation": th.nn.Tanh},
        ]
    ),
    net_arch=dict(pi=[128, 128, 128], vf=[128, 128, 128]),
)

# PPO algorithm params
model = PPO(
    CnnPolicy,
    my_env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    n_steps=2560,
    batch_size=64,
    learning_rate=0.000125,
    gamma=0.9,
    ent_coef=7.5e-08,
    clip_range=0.3,
    n_epochs=5,
    device="cuda",
    gae_lambda=1.0,
    max_grad_norm=0.9,
    vf_coef=0.95,
    tensorboard_log=log_dir,
)

# Save a checkpoint policy in the same log directory
checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=log_dir, name_prefix="jetbot_policy_checkpoint")

# Train the model
total_timesteps = 500000
model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

# Save the final policy
model.save(log_dir + "/jetbot_policy")

# Close the environment
my_env.close()
