from stable_baselines3 import PPO
from jetbot_env import HelloWorld
from omni.isaac.kit import SimulationApp

policy_path = "./cnn_policy/jetbot_policy.zip"

# Initialize the SimulationApp
simulation_app = SimulationApp({"headless": False, "anti_aliasing": 0})

# Create an instance of your JetBotEnv
my_env = HelloWorld(headless=False)  # Adjust the arguments as per your environment

# Load the pre-trained PPO model
model = PPO.load(policy_path)

# Perform evaluation episodes
for _ in range(20):
    my_env.reset()
    done = False
    while not done:
        actions, _ = model.predict(my_env.get_observation(), deterministic=True)
        _, reward, done, _ = my_env.step(actions)
        my_env.render()

# Close the environment and the SimulationApp
my_env.close()
simulation_app.close()