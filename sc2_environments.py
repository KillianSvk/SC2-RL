import math
import os

import numpy as np
import matplotlib.pyplot as plt
from absl import logging, flags
from enum import Enum

import cv2
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.env_checker import check_env

from pysc2.env import sc2_env
from pysc2.lib import actions, features, units, portspicker
from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Agent

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
FLAGS = flags.FLAGS
FLAGS(["run.py"])

FUNCTIONS = actions.FUNCTIONS
_PLAYER = 1
_NEUTRAL = 3


class SC2GymEnvironment(gym.Env):
    """ Wraps PySC2's SC2Env to make it Gym-compatible. """

    def __init__(self):
        super().__init__()

        self.obs = None
        self.sc2_env = None
        self.episodes = 0
        self.steps = 0

        self.screen_size = 32
        self.minimap_size = 32

        self.selected_marine = None
        self.GRID_SIZE = 11
        self.GRID_HALF_SIZE = self.GRID_SIZE // 2

        self.name = f"local_grid_{self.GRID_SIZE}x{self.GRID_SIZE}"

        self.init_sc2_env()
        self.define_action_space()
        self.define_observation_space()

    def __str__(self):
        return f"{self.name}"
    
    def init_sc2_env(self):
        self.sc2_env = SC2Env(
            map_name="CollectMineralShards",
            players=[Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=self.screen_size, minimap=self.minimap_size),
                use_feature_units=True,
                use_camera_position=True,
                crop_to_playable_area=True
            ),
            game_steps_per_episode=0,
            step_mul=8,
            realtime=False,
            visualize=False
        )

        self.obs = self.sc2_env.reset()[0]

    def define_action_space(self):
        """Defines the Gym-compatible action space for PySC2."""

        actions_num = (self.GRID_SIZE * self.GRID_SIZE)

        # actions_num = (len(self.obs.observation.available_actions))  # Number of possible actions in PySC2

        self.action_space = spaces.Discrete(actions_num)

    def define_observation_space(self):
        """Defines the Gym-compatible observation space."""

        observation_space = spaces.Box(low=-1, high=1, shape=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        # observation_space = spaces.Box(low=-1, high=1, shape=(self.GRID_SIZE * self.GRID_SIZE, ), dtype=np.int8)

        self.observation_space = observation_space

    def xy_locations(self, mask):
        """Mask should be a set of bools from comparison with a feature layer."""
        y, x = mask.nonzero()
        return list(zip(x, y))

    def is_in_local_grid(self, x, y) -> bool:
        selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y

        if x < selected_marine_x - self.GRID_HALF_SIZE or x > selected_marine_x + self.GRID_HALF_SIZE:
            return False

        if y < selected_marine_y - self.GRID_HALF_SIZE or y > selected_marine_y + self.GRID_HALF_SIZE:
            return False

        return True

    def is_in_bounds(self, x, y) -> bool:
        if x < 0 or x >= self.screen_size:
            return False

        if y < 0 or y >= self.screen_size:
            return False

        return True

    def global_to_local_pos(self, x, y) -> tuple[int, int]:
        selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y
        local_x = x - (selected_marine_x - self.GRID_HALF_SIZE)
        local_y = y - (selected_marine_y - self.GRID_HALF_SIZE)

        return local_x, local_y

    def get_gym_observation(self):
        selected_units = [unit for unit in self.obs.observation.feature_units if unit.is_selected]

        if len(selected_units) > 0:
            self.selected_marine = selected_units[0]

        screen = self.obs.observation.feature_screen.player_relative
        local_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        mineral_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        bounds_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        pathable_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)

        if self.selected_marine is not None:
            selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y

            # assign out of bounds
            for local_y in range(-self.GRID_HALF_SIZE, self.GRID_HALF_SIZE + 1):
                for local_x in range(-self.GRID_HALF_SIZE, self.GRID_HALF_SIZE + 1):
                    global_x, global_y = selected_marine_x + local_x, selected_marine_y + local_y
                    if not self.is_in_bounds(global_x, global_y):
                        bounds_grid[local_y + self.GRID_HALF_SIZE, local_x + self.GRID_HALF_SIZE] = 1

            # assign pathable
            pathable_screen = self.obs.observation.feature_screen.pathable
            not_pathable, pathable, mineral = -1, 0, 1
            for local_y in range(-self.GRID_HALF_SIZE, self.GRID_HALF_SIZE + 1):
                for local_x in range(-self.GRID_HALF_SIZE, self.GRID_HALF_SIZE + 1):
                    global_x, global_y = selected_marine_x + local_x, selected_marine_y + local_y
                    if self.is_in_bounds(global_x, global_y):
                        pathable_grid[local_y, local_x] = pathable_screen[global_y, global_x]
                        local_grid[local_y, local_x] = pathable if pathable_screen[global_y, global_x] == 1 else not_pathable

                    else:
                        pathable_grid[local_y, local_x] = 0
                        local_grid[local_y, local_x] = not_pathable

            # assign minerals
            minerals = self.xy_locations(screen == _NEUTRAL)
            for x, y in minerals:
                if self.is_in_local_grid(x, y):
                    local_x, local_y = self.global_to_local_pos(x, y)
                    mineral_grid[local_y, local_x] = 1
                    local_grid[local_y, local_x] = mineral


        # print("-------------------------")
        # for line in local_grid.tolist():
        #     print(line)

        return local_grid

    def local_to_global_action(self, action) -> tuple[int, int]:
        if self.selected_marine is None:
            return 0, 0

        selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y
        local_x, local_y = action % self.GRID_SIZE, action // self.GRID_SIZE
        local_x, local_y = local_x - self.GRID_HALF_SIZE, local_y - self.GRID_HALF_SIZE
        global_x, global_y = local_x + selected_marine_x, local_y + selected_marine_y

        return global_x, global_y

    def reward_func(self) -> int:
        reward = self.obs.reward
        reward -= 0.01

        return reward

    def get_info(self):
        info = {}

        score = self.obs.observation['score_cumulative']['score']
        info["score"] = score

        return info

    def step(self, action):
        """Executes the given action in the SC2 environment and returns the new state, reward, and episode status."""
        x, y = self.local_to_global_action(action)

        player_relative = self.obs.observation.feature_screen.player_relative
        available_actions = self.obs.observation.available_actions

        if FUNCTIONS.Move_screen.id in available_actions:
            if self.is_in_bounds(x, y):
                sc2_action = [FUNCTIONS.Move_screen("now", (x, y))]

            else:
                sc2_action = [FUNCTIONS.no_op()]

        else:
            # sc2_action = [FUNCTIONS.select_army("select")]
            marines_pos = self.xy_locations(player_relative == _PLAYER)
            marine_pos = marines_pos[0]
            sc2_action = [FUNCTIONS.select_point("select", marine_pos)]

        time_step = self.sc2_env.step(sc2_action)

        # Unpack the returned values
        self.obs = time_step[0]
        reward = self.reward_func()
        done = self.obs.last()

        self.steps += 1

        # Truncated flag (e.g., if the episode ended due to time constraints)
        truncated = False

        # Info dictionary (optional debug info)
        info = {}
        obs = self.get_gym_observation()

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        self.episodes += 1

        self.obs = self.sc2_env.reset()[0]
        gym_observation = self.get_gym_observation()
        info = self.get_info()

        return gym_observation, info

    def render(self, mode="human"):
        """ Optional: Implement visualization. """
        pass

    def close(self):
        """ Close the SC2 environment. """
        self.sc2_env.close()


class SC2FlattenEnv(SC2GymEnvironment):
    def __init__(self):
        super(SC2FlattenEnv, self).__init__()

        self.GRID_SIZE = 11
        self.GRID_HALF_SIZE = self.GRID_SIZE // 2
        self.name = f"flatten_obs_env_{self.GRID_SIZE}x{self.GRID_SIZE}"

        self.define_action_space()
        self.define_observation_space()

    def define_action_space(self):
        self.action_space = spaces.Discrete(self.GRID_SIZE * self.GRID_SIZE)

    def define_observation_space(self):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.GRID_SIZE * self.GRID_SIZE, ), dtype=np.int8)

    def get_gym_observation(self):
        gym_obs = super(SC2FlattenEnv, self).get_gym_observation()
        flatten_gym_obs = gym_obs.flatten()

        return flatten_gym_obs


class SC2DirectionActionsEnv(SC2GymEnvironment):
    class Direction(Enum):
        N = 0
        NE = 1
        E = 2
        SE = 3
        S = 4
        SW = 5
        W = 6
        NW = 7

    def __init__(self):
        super().__init__()

        self.name = "direction_actions"

    def define_action_space(self):
        self.action_space = spaces.Discrete(len(self.Direction))

    def define_observation_space(self):
        self.observation_space = spaces.Box(low=0, high=4, shape=(3, self.minimap_size, self.minimap_size), dtype=np.uint8)

    def get_gym_observation(self):
        selected = self.obs.observation.feature_screen.selected
        player_relative = self.obs.observation.feature_screen.player_relative
        pathable = self.obs.observation.feature_screen.pathable

        observation = np.array([selected, player_relative, pathable], dtype=np.uint8)

        return observation

    def direction_to_action(self, direction):
        movement_distance = 5
        selected_units = [unit for unit in self.obs.observation.feature_units if unit.is_selected]

        if len(selected_units) > 0:
            self.selected_marine = selected_units[0]

        if self.selected_marine is not None:
            selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y

            match direction:
                case self.Direction.N.value:
                    new_x, new_y = selected_marine_x, selected_marine_y + movement_distance

                case self.Direction.NE.value:
                    new_x, new_y = selected_marine_x + movement_distance/math.sqrt(2), selected_marine_y + movement_distance/math.sqrt(2)

                case self.Direction.E.value:
                    new_x, new_y = selected_marine_x + movement_distance, selected_marine_y

                case self.Direction.SE.value:
                    new_x, new_y = selected_marine_x + movement_distance/math.sqrt(2), selected_marine_y - movement_distance/math.sqrt(2)

                case self.Direction.S.value:
                    new_x, new_y = selected_marine_x, selected_marine_y - movement_distance

                case self.Direction.SW.value:
                    new_x, new_y = selected_marine_x - movement_distance/math.sqrt(2), selected_marine_y - movement_distance/math.sqrt(2)

                case self.Direction.W.value:
                    new_x, new_y = selected_marine_x - movement_distance, selected_marine_y

                case self.Direction.NW.value:
                    new_x, new_y = selected_marine_x - movement_distance/math.sqrt(2), selected_marine_y + movement_distance/math.sqrt(2)

                case _:
                    raise Exception("Invalid action, doesn't correspond to any direction")

            if self.is_in_bounds(new_x, new_y):
                return FUNCTIONS.Move_screen("now", (new_x, new_y))

        return FUNCTIONS.no_op()

    def step(self, direction):
        player_relative = self.obs.observation.feature_screen.player_relative
        available_actions = self.obs.observation.available_actions

        if FUNCTIONS.Move_screen.id in available_actions:
            sc2_action = [self.direction_to_action(direction)]

        else:
            # sc2_action = [FUNCTIONS.select_army("select")]
            marines_pos = self.xy_locations(player_relative == _PLAYER)
            marine_pos = marines_pos[0]
            sc2_action = [FUNCTIONS.select_point("select", marine_pos)]

        time_step = self.sc2_env.step(sc2_action)

        self.obs = time_step[0]

        obs = self.get_gym_observation()
        reward = self.obs.reward
        done = self.obs.last()
        truncated = False
        info = {}

        return obs, reward, done, truncated, info


class SC2SmallActionEnv(SC2DirectionActionsEnv):
    def __init__(self):
        super().__init__()

        self.GRID_SIZE = 11
        self.GRID_HALF_SIZE = self.GRID_SIZE // 2
        self.name = f"loca_grid_{self.GRID_SIZE}x{self.GRID_SIZE}_small_action"

        self.define_observation_space()

    def define_observation_space(self):
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

    def get_gym_observation(self):
        screen = self.obs.observation.feature_screen.player_relative
        local_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        selected_units = [unit for unit in self.obs.observation.feature_units if unit.is_selected]

        if len(selected_units) > 0:
            self.selected_marine = selected_units[0]

        if self.selected_marine is not None:
            selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y

            # assign out of bounds to local grid
            for local_y in range(-self.GRID_HALF_SIZE, self.GRID_HALF_SIZE + 1):
                for local_x in range(-self.GRID_HALF_SIZE, self.GRID_HALF_SIZE + 1):
                    global_x, global_y = selected_marine_x + local_x, selected_marine_y + local_y
                    if not self.is_in_bounds(global_x, global_y):
                        local_grid[local_y + self.GRID_HALF_SIZE, local_x + self.GRID_HALF_SIZE] = -1

            # assign minerals to local grid
            minerals = self.xy_locations(screen == _NEUTRAL)
            for x, y in minerals:
                if self.is_in_local_grid(x, y):
                    local_x, local_y = self.global_to_local_pos(x, y)
                    local_grid[local_y, local_x] = 1

        return local_grid


class SC2ScreenEnv(SC2GymEnvironment):
    def __init__(self):
        self.obs = None
        self.sc2_env = None
        self.episodes = 0
        self.steps = 0

        self.screen_size = 36
        self.minimap_size = 36
        self.selected_marine = None
        self.name = f"screen_{self.screen_size}x{self.screen_size}"

        self.movement_distance = 5
        SQRT2_INV = 1 / math.sqrt(2)
        self.move_deltas = [
            (0, -1),  # up
            (0, 1),  # down
            (-1, 0),  # left
            (1, 0),  # right
            (-SQRT2_INV, -SQRT2_INV),  # up-left
            (SQRT2_INV, -SQRT2_INV),  # up-right
            (-SQRT2_INV, SQRT2_INV),  # down-left
            (SQRT2_INV, SQRT2_INV),  # down-right
        ]

        self.init_sc2_env()
        self.define_action_space()
        self.define_observation_space()

    def init_sc2_env(self):
        self.sc2_env = SC2Env(
            map_name="CollectMineralShards",
            players=[Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=self.screen_size, minimap=self.minimap_size),
                rgb_dimensions=features.Dimensions(screen=self.screen_size, minimap=self.minimap_size),
                use_feature_units=True,
                use_camera_position=True,
                crop_to_playable_area=True,
                action_space=actions.ActionSpace.FEATURES,
            ),
            game_steps_per_episode=0,
            step_mul=8,
            realtime=False,
            visualize=False
        )

        self.obs = self.sc2_env.reset()[0]

    def define_action_space(self):
        self.action_space = spaces.Discrete(len(self.move_deltas))

    def define_observation_space(self):
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, self.screen_size, self.screen_size), dtype=np.uint8)

    def get_gym_observation(self):
        selected_units = [unit for unit in self.obs.observation.feature_units if unit.is_selected]

        if len(selected_units) > 0:
            self.selected_marine = selected_units[0]

        # bgr_screen = self.obs.observation["rgb_minimap"]
        # rgb_screen = bgr_screen[..., ::-1]
        #
        # rgb_screen_uint8 = rgb_screen.astype(np.uint8)
        # rgb_screen_transposed = np.transpose(rgb_screen_uint8, (2, 0, 1))

        minimap_obs = np.full(shape=(3, self.screen_size, self.screen_size), fill_value=0, dtype=np.uint8)
        minimap_player_relative = self.obs.observation.feature_minimap["player_relative"]
        minimap_selected = self.obs.observation.feature_minimap["selected"]

        player_relative_xy = self.xy_locations(minimap_player_relative == _NEUTRAL)
        selected_xy = self.xy_locations(minimap_selected == 1)

        for x, y in selected_xy:
            lime_green = [50, 205, 50]
            minimap_obs[:, y, x] = lime_green

        for x, y in player_relative_xy:
            light_blue = [173, 216, 230]
            minimap_obs[:, y, x] = light_blue

        # plt.imshow(minimap_obs.transpose((1, 2, 0)))
        # plt.title("RGB Screen Observation")
        # plt.axis('off')
        # plt.show()

        return minimap_obs

    def step(self, action):
        player_relative = self.obs.observation.feature_screen.player_relative
        available_actions = self.obs.observation.available_actions

        if FUNCTIONS.Move_screen.id in available_actions:
            dx, dy = self.move_deltas[action]
            dx, dy = self.movement_distance * dx, self.movement_distance * dy
            x, y = self.selected_marine.x + dx, self.selected_marine.y + dy

            if self.is_in_bounds(x, y):
                sc2_action = [FUNCTIONS.Move_screen("now", (x, y))]

            else:
                sc2_action = [FUNCTIONS.no_op()]

        else:
            marines_pos = self.xy_locations(player_relative == _PLAYER)
            marine_pos = marines_pos[0]
            sc2_action = [FUNCTIONS.select_point("select", marine_pos)]

        time_step = self.sc2_env.step(sc2_action)

        # Unpack the returned values
        self.obs = time_step[0]
        self.steps += 1

        obs = self.get_gym_observation()
        reward = self.reward_func()
        done = self.obs.last()
        truncated = False
        info = self.get_info()

        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        obs = self.get_gym_observation()  # This already includes the custom background
        frame = obs.transpose((1, 2, 0))  # CHW → HWC for OpenCV

        old_background_color = [0, 0, 0]
        new_background_color = [20, 100, 20]
        mask = cv2.inRange(frame, np.array(old_background_color), np.array(old_background_color))
        frame[mask > 0] = new_background_color

        scale_factor = 20
        enlarged_frame = cv2.resize(
            frame,
            (self.screen_size * scale_factor, self.screen_size * scale_factor),
            interpolation=cv2.INTER_NEAREST
        )

        # Convert RGB to BGR for OpenCV display
        frame_bgr = cv2.cvtColor(enlarged_frame, cv2.COLOR_RGB2BGR)

        # Show the image
        cv2.imshow("Minimap Observation", frame_bgr)
        cv2.waitKey(1)  # Refresh window



if __name__ == "__main__":
    test_env = None

    try:
        test_env = SC2ScreenEnv()
        check_env(test_env)

    finally:
        if test_env is not None:
            test_env.close()
