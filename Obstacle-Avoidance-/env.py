import gym
from gym import spaces
import numpy as np
import math
from omni.isaac.kit import SimulationApp
from omni.isaac.core import World
from omni.isaac.wheeled_robots.robots import WheeledRobot
from omni.isaac.wheeled_robots.controllers.differential_controller import DifferentialController
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.core.utils.nucleus import get_assets_root_path

class JetBotEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=256,
        seed=0,
        headless=True,
        camera=True,  # Set the camera parameter to True by default
    ) -> None:
        try:
            self._simulation_app = SimulationApp({"headless": headless, "anti_aliasing": 0})
        except Exception as e:
            print(f"Error initializing SimulationApp: {e}")
            raise e

        self.headless = headless
        self.camera = camera  # Store the camera parameter
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)

        try:
            self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=1.0)
            self._my_world.scene.add_default_ground_plane()
        except Exception as e:
            print(f"Error initializing World: {e}")
            self._simulation_app.close()
            raise e

        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            print("Could not find Isaac Sim assets folder")
            self._simulation_app.close()
            return

        jetbot_asset_path = assets_root_path + "/Isaac/Robots/Jetbot/jetbot.usd"
        try:
            self.jetbot = self._my_world.scene.add(
                WheeledRobot(
                    prim_path="/jetbot",
                    name="my_jetbot",
                    wheel_dof_names=["left_wheel_joint", "right_wheel_joint"],
                    create_robot=True,
                    usd_path=jetbot_asset_path,
                    position=np.array([0, 0.0, 0.0]),
                    orientation=np.array([1.0, 0.0, 0.0, 0.0]),
                    camera=camera,  # Pass the camera parameter to WheeledRobot
                )
            )
        except Exception as e:
            print(f"Error adding Jetbot to the scene: {e}")
            self._simulation_app.close()
            raise e

        self.jetbot_controller = DifferentialController(name="simple_control", wheel_radius=0.0325, wheel_base=0.1125)

        try:
            self.goal = self._my_world.scene.add(
                VisualCuboid(
                    prim_path="/new_cube_1",
                    name="visual_cube",
                    position=np.array([2.0, 0.0, 0.1]),
                    size=0.1,
                    color=np.array([1.0, 0, 0]),
                )
            )
        except Exception as e:
            print(f"Error adding goal to the scene: {e}")
            self._simulation_app.close()
            raise e

        self.seed(seed)
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)

        # Define camera parameters
        self.camera_image_shape = (256, 256, 3)  # Update with your camera resolution
        self.observation_space = self.define_observation_space()

        self.action_space = spaces.Box(low=-1, high=1.0, shape=(2,), dtype=np.float32)

        self.max_velocity = 1
        self.max_angular_velocity = math.pi
        self.reset_counter = 0
        return

    def define_observation_space(self):
        observation_space_dict = {
            'collision_reward': spaces.Box(low=float("-inf"), high=float("inf"), shape=(1,), dtype=np.float32),
            'other_info': spaces.Box(low=float("inf"), high=float("-inf"), shape=(16,), dtype=np.float32),
        }
        if self.camera:
            observation_space_dict['camera_image'] = spaces.Box(low=0, high=255, shape=self.camera_image_shape, dtype=np.uint8)
        return spaces.Dict(observation_space_dict)

    def get_dt(self):
        return self._dt

    def step(self, action):
        try:
            previous_jetbot_position, _ = self.jetbot.get_world_pose()
            raw_forward = action[0]
            raw_angular = action[1]

            forward = (raw_forward + 1.0) / 2.0
            forward_velocity = forward * self.max_velocity

            angular_velocity = raw_angular * self.max_angular_velocity

            for i in range(self._skip_frame):
                self.jetbot.apply_wheel_actions(
                    self.jetbot_controller.forward(command=[forward_velocity, angular_velocity])
                )
                self._my_world.step(render=False)

            # Compute collision reward based on the difference in distances to the goal before and after the action
            goal_world_position, _ = self.goal.get_world_pose()
            current_jetbot_position, _ = self.jetbot.get_world_pose()
            previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_jetbot_position)
            current_dist_to_goal = np.linalg.norm(goal_world_position - current_jetbot_position)
            collision_reward = previous_dist_to_goal - current_dist_to_goal

            # Penalize collisions
            if current_dist_to_goal < 0.1:
                collision_reward -= 1.0

            observations = self.get_observations()
            info = {}
            done = False
            if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
                done = True
            if current_dist_to_goal < 0.1:
                done = True

            return observations, collision_reward, done, info
        except Exception as e:
            print(f"Error in step method: {e}")
            self._simulation_app.close()
            raise e

    def reset(self):
        try:
            self._my_world.reset()
            self.reset_counter = 0
            alpha = 2 * math.pi * np.random.rand()
            r = 1.00 * math.sqrt(np.random.rand()) + 0.20
            self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, 0.05]))

            observations = self.get_observations()
            return observations
        except Exception as e:
            print(f"Error in reset method: {e}")
            self._simulation_app.close()
            raise e

    def get_observations(self):
        try:
            self._my_world.render()
            jetbot_world_position, jetbot_world_orientation = self.jetbot.get_world_pose()
            jetbot_linear_velocity = self.jetbot.get_linear_velocity()
            jetbot_angular_velocity = self.jetbot.get_angular_velocity()
            goal_world_position, _ = self.goal.get_world_pose()

            observation_dict = {
                'collision_reward': 0.0,
                'other_info': np.concatenate(
                    [
                        jetbot_world_position,
                        jetbot_world_orientation,
                        jetbot_linear_velocity,
                        jetbot_angular_velocity,
                        goal_world_position,
                    ]
                ),
            }
            if self.camera:
                observation_dict['camera_image'] = self.get_camera_image()

            return observation_dict
        except Exception as e:
            print(f"Error in get_observations method: {e}")
            self._simulation_app.close()
            raise e

    def get_camera_image(self):
        try:
            if self.camera:
                # Replace this with the actual code to get the camera image from the simulation
                # Assuming get_camera_rgb() method exists, update accordingly based on Isaac Sim API
                return self._my_world.physics_interfaces[0].get_camera_rgb()
            else:
                return None
        except Exception as e:
            print(f"Error in get_camera_image method: {e}")
            self._simulation_app.close()
            raise e

    def render(self, mode="human"):
        return

    def close(self):
        try:
            self._simulation_app.close()
        except Exception as e:
            print(f"Error closing the simulation app: {e}")

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]