import math
import os
from typing import SupportsFloat, Any

from stable_baselines3.common.env_checker import check_env

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from abc import ABC, abstractmethod
from absl import flags, logging

import cv2
import gymnasium as gym
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame

from pysc2.env.enums import Race
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FUNCTIONS
from pysc2.env.sc2_env import SC2Env, Agent

FLAGS = flags.FLAGS
FLAGS(["sc2_environments.py"])
logging.set_verbosity(logging.INFO)
logging.info("SC2Environments module initialized.")


_PLAYER = 1
_NEUTRAL = 3
_MINERAL_SHARD = 1680


class SC2GymWrapper(gym.Env, ABC):
    """SC2GymWrapper is an abstract base class that wraps PySC2's SC2Env to make it compatible with the Gymnasium API.

    This class provides a foundation for creating custom StarCraft II environments that can be used with reinforcement
    learning algorithms. It defines the structure and required methods for implementing Gym-compatible environments.

    Attributes:
        sc2_env (SC2Env): The underlying PySC2 environment instance.
        obs (Any): The current observation from the SC2 environment.
        episodes (int): The number of episodes completed in the environment.
        steps (int): The number of steps taken in the current episode.
        screen_size (int): The size of the screen observation space.
        minimap_size (int): The size of the minimap observation space.

    Abstract Properties:
        action_space (gym.Space): Defines the Gym-compatible action space for the environment.
        observation_space (gym.Space): Defines the Gym-compatible observation space for the environment.
        name (str): The name of the environment implementation.

    Abstract Methods:
        get_gym_observation(self):
            Returns an observation from pysc2 environment in Gym-compatible space

        step(action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
            Executes the given action in the environment and returns the new state, reward, and episode status.

    Methods:
        reset(seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[ObsType, dict[str, Any]]:
            Resets the environment to its initial state and returns the initial observation and optional info.

        render()

        close():
            Frees any resources used by the environment.

        in_screen_bounds(self, x, y) -> bool:
            Return True if position in inside the screen bounds

        xy_locations(mask: np.ndarray) -> list[tuple[int, int]]:
            Converts a boolean mask into a list of (x, y) coordinates where the mask is True.

    Usage:
        This class is intended to be subclassed to create specific SC2 environments. Subclasses must implement the
        abstract properties and methods to define the action space, observation space, and environment behavior.
    """

    def __init__(self, screen_size, minimap_size):
        self.episodes = 0
        self.steps = 0

        self.screen_size = screen_size
        self.minimap_size = minimap_size

        self.obs = None
        self.sc2_env = None

        self.init_sc2_env()

    def __str__(self):
        return f"{self.name}"

    @property
    def obs(self):
        """Current observations from PySC2 environment."""
        return self._obs

    @obs.setter
    def obs(self, value):
        """Sets the current observations for the SC2 environment."""
        self._obs = value

    @property
    @abstractmethod
    def action_space(self):
        """Defines the Gym-compatible action space."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def observation_space(self):
        """Defines the Gym-compatible observation space."""
        raise NotImplementedError()

    @property
    @abstractmethod
    def name(self):
        """Name of implementation and a name under which it will be saved."""
        raise NotImplementedError()

    @abstractmethod
    def get_gym_observation(self):
        """Returns an observation from pysc2 environment in Gym-compatible space"""
        raise NotImplementedError()

    @abstractmethod
    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        raise NotImplementedError()

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None, ) -> tuple[ObsType, dict[str, Any]]:
        self.steps = 0
        self.episodes += 1
        self.obs = self.sc2_env.reset()[0]

        gym_observation = self.get_gym_observation()
        info = {}

        return gym_observation, info

    def close(self):
        """Frees any resources used by the environment. """
        self.sc2_env.close()

    def init_sc2_env(self):
        """Initializes the StarCraft II (SC2) environment.

        This method sets up the SC2 environment using the `SC2Env` class from PySC2. It configures the environment with
        specific parameters such as the map name, player race, agent interface format, and other gameplay settings.
        The initialized environment is stored in the `self.sc2_env` attribute, and the initial observation is stored
        in the `self.obs` attribute.

        Attributes Set:
            - `self.sc2_env`: The initialized SC2 environment instance.
            - `self.obs`: The initial observation from the SC2 environment."""

        sc2_env = SC2Env(
            map_name="CollectMineralShards",
            players=[Agent(Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=self.screen_size, minimap=self.minimap_size),
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

        self.sc2_env = sc2_env
        self.obs = sc2_env.reset()[0]

    @abstractmethod
    def render(self) -> RenderFrame | list[RenderFrame] | None:
        """Renders the current state of the environment.

        This method is responsible for visualizing the environment's state. The implementation
        should provide a way to display the environment's current observation, which can be useful
        for debugging or monitoring the agent's performance.

        Returns:
            RenderFrame | list[RenderFrame] | None: A rendered frame or a list of frames, or None if rendering is not implemented.
        """
        pass

    def in_screen_bounds(self, x, y) -> bool:
        return 0 <= x < self.screen_size and 0 <= y < self.screen_size

    @staticmethod
    def xy_locations(mask):
        """Converts a boolean mask into a list of (x, y) coordinates where the mask is True."""
        y, x = mask.nonzero()
        return list(zip(x, y))


class SC2LocalObservationEnv(SC2GymWrapper):

    def __init__(self, grid_size=11):
        super().__init__(32, 32)

        self.selected_marine = None
        self.GRID_SIZE = grid_size
        self.GRID_HALF_SIZE = self.GRID_SIZE // 2

    @property
    def name(self):
        return f"local_grid_{self.GRID_SIZE}x{self.GRID_SIZE}"

    @property
    def observation_space(self):
        observation_space = spaces.Box(low=-1, high=1, shape=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        # observation_space = spaces.Box(low=-1, high=1, shape=(self.GRID_SIZE * self.GRID_SIZE, ), dtype=np.int8)

        return observation_space

    @property
    def action_space(self):
        actions_count = (self.GRID_SIZE * self.GRID_SIZE)

        action_space = spaces.Discrete(actions_count)

        return action_space

    def is_in_local_grid(self, x, y) -> bool:
        selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y

        if x < selected_marine_x - self.GRID_HALF_SIZE or x > selected_marine_x + self.GRID_HALF_SIZE:
            return False

        if y < selected_marine_y - self.GRID_HALF_SIZE or y > selected_marine_y + self.GRID_HALF_SIZE:
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
                    if not self.in_screen_bounds(global_x, global_y):
                        bounds_grid[local_y + self.GRID_HALF_SIZE, local_x + self.GRID_HALF_SIZE] = 1

            # assign pathable
            pathable_screen = self.obs.observation.feature_screen.pathable
            not_pathable, pathable, mineral = -1, 0, 1
            for local_y in range(-self.GRID_HALF_SIZE, self.GRID_HALF_SIZE + 1):
                for local_x in range(-self.GRID_HALF_SIZE, self.GRID_HALF_SIZE + 1):
                    global_x, global_y = selected_marine_x + local_x, selected_marine_y + local_y
                    if self.in_screen_bounds(global_x, global_y):
                        pathable_grid[local_y, local_x] = pathable_screen[global_y, global_x]
                        local_grid[local_y, local_x] = pathable if pathable_screen[
                                                                       global_y, global_x] == 1 else not_pathable

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

    def get_step_info(self):
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
            if self.in_screen_bounds(x, y):
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
        info = self.get_step_info()
        obs = self.get_gym_observation()

        return obs, reward, done, truncated, info

    def reset(self, seed=None, options=None):
        self.episodes += 1

        self.obs = self.sc2_env.reset()[0]
        gym_observation = self.get_gym_observation()
        info = self.get_step_info()

        return gym_observation, info

    def render(self, mode="human"):
        pass


class SC2LocalRoomsEnv(SC2LocalObservationEnv):

    @property
    def name(self):
        return f"rooms_local_grid_{self.GRID_SIZE}x{self.GRID_SIZE}"

    @property
    def observation_space(self):
        observation_space = spaces.Box(low=0, high=1, shape=(2, self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)

        return observation_space

    def init_sc2_env(self):
        sc2_env = SC2Env(
            map_name="CollectMineralShardsRooms",
            players=[Agent(Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=self.screen_size, minimap=self.minimap_size),
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

        self.sc2_env = sc2_env
        self.obs = sc2_env.reset()[0]

    def get_gym_observation(self):
        selected_units = [unit for unit in self.obs.observation.feature_units if unit.is_selected]

        if len(selected_units) > 0:
            self.selected_marine = selected_units[0]

        screen = self.obs.observation.feature_screen.player_relative

        bounds_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        pathable_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        mineral_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)

        if self.selected_marine is not None:
            selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y

            # # assign out of bounds
            # for local_y in range(-self.GRID_HALF_SIZE, self.GRID_HALF_SIZE + 1):
            #     for local_x in range(-self.GRID_HALF_SIZE, self.GRID_HALF_SIZE + 1):
            #         global_x, global_y = selected_marine_x + local_x, selected_marine_y + local_y
            #         if not self.in_screen_bounds(global_x, global_y):
            #             bounds_grid[local_y + self.GRID_HALF_SIZE, local_x + self.GRID_HALF_SIZE] = 1

            # assign pathable
            pathable_screen = self.obs.observation.feature_screen.pathable
            for local_y in range(-self.GRID_HALF_SIZE, self.GRID_HALF_SIZE + 1):
                for local_x in range(-self.GRID_HALF_SIZE, self.GRID_HALF_SIZE + 1):
                    global_x, global_y = selected_marine_x + local_x, selected_marine_y + local_y
                    if self.in_screen_bounds(global_x, global_y):
                        pathable_grid[local_y, local_x] = pathable_screen[global_y, global_x]

                    else:
                        pathable_grid[local_y, local_x] = 0

            # assign minerals
            minerals = self.xy_locations(screen == _NEUTRAL)
            for x, y in minerals:
                if self.is_in_local_grid(x, y):
                    local_x, local_y = self.global_to_local_pos(x, y)
                    mineral_grid[local_y, local_x] = 1

        return np.array([pathable_grid, mineral_grid])


class SC2LocalObservationFlattenedEnv(SC2LocalObservationEnv):
    def __init__(self):
        super().__init__()

    @property
    def name(self):
        return f"local_grid_flattened_env_{self.GRID_SIZE}x{self.GRID_SIZE}"

    @property
    def action_space(self):
        return spaces.Discrete(self.GRID_SIZE * self.GRID_SIZE)

    @property
    def observation_space(self):
        return spaces.Box(low=-1, high=1, shape=(self.GRID_SIZE * self.GRID_SIZE,), dtype=np.int8)

    def get_gym_observation(self):
        gym_obs = super(SC2LocalObservationFlattenedEnv, self).get_gym_observation()
        flatten_gym_obs = gym_obs.flatten()

        return flatten_gym_obs


class SC2ScreenEnv(SC2GymWrapper):

    def __init__(self):
        super().__init__(36, 36)

        self.selected_marine = None

        self.movement_distance = 5
        self.move_deltas = self.define_move_deltas()

    @property
    def name(self):
        return f"screen_{self.screen_size}x{self.screen_size}"

    @property
    def action_space(self):
        return spaces.Discrete(len(self.move_deltas))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=255, shape=(3, self.screen_size, self.screen_size), dtype=np.uint8)

    @staticmethod
    def define_move_deltas():
        SQRT2_INV = 1 / math.sqrt(2)

        # move_deltas = [
        #     (0, 0),
        #     (0, -1),  # up
        #     (0, 1),  # down
        #     (-1, 0),  # left
        #     (1, 0),  # right
        #     (-SQRT2_INV, -SQRT2_INV),  # up-left
        #     (SQRT2_INV, -SQRT2_INV),  # up-right
        #     (-SQRT2_INV, SQRT2_INV),  # down-left
        #     (SQRT2_INV, SQRT2_INV),  # down-right
        # ]

        move_deltas = [
            (0, 0),
            (0, -1),  # up
            (0, 1),  # down
            (-1, 0),  # left
            (1, 0),  # right
            (-1, -1),  # up-left
            (1, -1),  # up-right
            (-1, 1),  # down-left
            (1, 1),  # down-right
        ]

        return move_deltas

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

    def reward_func(self) -> int:
        reward = self.obs.reward
        reward -= 0.01

        return reward

    def step(self, action):
        player_relative = self.obs.observation.feature_screen.player_relative
        available_actions = self.obs.observation.available_actions

        if FUNCTIONS.Move_screen.id in available_actions:
            dx, dy = self.move_deltas[action]
            dx, dy = self.movement_distance * dx, self.movement_distance * dy
            x, y = self.selected_marine.x + dx, self.selected_marine.y + dy

            if self.in_screen_bounds(x, y):
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

        info = {}
        score = self.obs.observation['score_cumulative']['score']
        info["score"] = score

        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        obs = self.get_gym_observation()
        frame = obs.transpose((1, 2, 0))
        scale_factor = 20
        enlarged_frame = cv2.resize(frame, (self.screen_size * scale_factor, self.screen_size * scale_factor),
                                    interpolation=cv2.INTER_NEAREST)
        frame_bgr = cv2.cvtColor(enlarged_frame, cv2.COLOR_RGB2BGR)

        game_loop = self.obs.observation.game_loop[0]
        seconds = game_loop / 22.4
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        cv2.putText(
            frame_bgr,
            f"Time: {time_str}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        score = self.obs.observation['score_cumulative']['score']
        cv2.putText(
            frame_bgr,
            f"Score: {score}",
            (400, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Minimap Observation", frame_bgr)
        cv2.waitKey(1)


class SC2ScreenBoxEnv(SC2ScreenEnv):

    @property
    def name(self):
        return f"screen_box_{self.screen_size}x{self.screen_size}"

    @property
    def action_space(self):
        return spaces.Box(low=-1.0, high=1.0, shape=(2,))

    def step(self, action):
        player_relative = self.obs.observation.feature_screen.player_relative
        available_actions = self.obs.observation.available_actions

        if FUNCTIONS.Move_screen.id in available_actions:
            dx, dy = action
            dx, dy = int(round(dx)), int(round(dy))
            dx, dy = self.movement_distance * dx, self.movement_distance * dy
            x, y = self.selected_marine.x + dx, self.selected_marine.y + dy

            if self.in_screen_bounds(x, y):
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

        info = {}
        score = self.obs.observation['score_cumulative']['score']
        info["score"] = score

        return obs, reward, done, truncated, info


class SC2MiddleInvisibleEnv(SC2GymWrapper):
    def __init__(self):
        super().__init__(24, 24)

        self.selected_marine = None

        self.GRID_SIZE = 48
        self.GRID_HALF_SIZE = self.GRID_SIZE // 2

        self.movement_distance = 3
        self.move_deltas = SC2ScreenEnv.define_move_deltas()

    @property
    def action_space(self):
        return spaces.Discrete(len(self.move_deltas))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=255, shape=(3, self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)

    @property
    def name(self):
        return f"rooms_middle_invisible_{self.GRID_SIZE}x{self.GRID_SIZE}"

    def init_sc2_env(self):
        sc2_env = SC2Env(
            map_name="CollectMineralShardsRooms",
            players=[Agent(Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=self.screen_size, minimap=self.minimap_size),
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

        self.sc2_env = sc2_env
        self.obs = sc2_env.reset()[0]

    def global_to_local_pos(self, x, y) -> tuple[int, int]:
        selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y
        local_x = x - (selected_marine_x - self.GRID_HALF_SIZE)
        local_y = y - (selected_marine_y - self.GRID_HALF_SIZE)

        return local_x, local_y

    def is_in_local_grid(self, x, y) -> bool:
        return 0 <= x < self.GRID_SIZE and 0 <= y < self.GRID_SIZE

    def get_gym_observation(self):
        selected_units = [unit for unit in self.obs.observation.feature_units if unit.is_selected]

        if len(selected_units) > 0:
            self.selected_marine = selected_units[0]

        minimap_obs = np.full(shape=(3, self.GRID_SIZE, self.GRID_SIZE), fill_value=100, dtype=np.uint8)

        mineral = [173, 216, 230]
        selected = [50, 205, 50]
        pathable = [255, 255, 255]
        not_pathable = [255, 0, 50]

        if self.selected_marine is not None:
            mineral_shards = [unit for unit in self.obs.observation.feature_units if unit.unit_type == _MINERAL_SHARD]
            minimap_pathable = self.obs.observation.feature_minimap["pathable"]

            xy_pathable = self.xy_locations(minimap_pathable == 1)

            for x, y in xy_pathable:
                local_x, local_y = self.global_to_local_pos(x, y)
                if self.is_in_local_grid(local_x, local_y):
                    minimap_obs[:, local_y, local_x] = pathable

            for mineral_shard in mineral_shards:
                x, y = mineral_shard.x, mineral_shard.y
                local_x, local_y = self.global_to_local_pos(x, y)
                if self.is_in_local_grid(local_x, local_y):
                    minimap_obs[:, local_y, local_x] = mineral

            # selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y
            # local_x, local_y = self.global_to_local_pos(selected_marine_x, selected_marine_y)
            # minimap_obs[:, local_y, local_x] = selected

        return minimap_obs

    def reward_func(self) -> int:
        reward = self.obs.reward
        reward -= 0.01

        return reward

    def get_step_info(self) -> dict[str, Any]:
        info = dict()
        score = self.obs.observation['score_cumulative']['score']
        info["score"] = score

        return info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        player_relative = self.obs.observation.feature_screen.player_relative
        available_actions = self.obs.observation.available_actions

        if FUNCTIONS.Move_screen.id in available_actions:
            dx, dy = self.move_deltas[action]
            dx, dy = self.movement_distance * dx, self.movement_distance * dy
            x, y = self.selected_marine.x + dx, self.selected_marine.y + dy

            if self.in_screen_bounds(x, y):
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

        info = self.get_step_info()

        return obs, reward, done, truncated, info

    def render(self, mode="human"):
        obs = self.get_gym_observation()
        frame = obs.transpose((1, 2, 0))
        scale_factor = 20
        enlarged_frame = cv2.resize(frame, (self.screen_size * scale_factor, self.screen_size * scale_factor),
                                    interpolation=cv2.INTER_NEAREST)
        frame_bgr = cv2.cvtColor(enlarged_frame, cv2.COLOR_RGB2BGR)

        game_loop = self.obs.observation.game_loop[0]
        seconds = game_loop / 22.4
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        time_str = f"{minutes:02d}:{seconds:02d}"

        cv2.putText(
            frame_bgr,
            f"Time: {time_str}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        score = self.obs.observation['score_cumulative']['score']
        cv2.putText(
            frame_bgr,
            f"Score: {score}",
            (250, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )

        cv2.imshow("Minimap Observation", frame_bgr)
        cv2.waitKey(1)


class SC2MiddleVisibleEnv(SC2MiddleInvisibleEnv):
    @property
    def name(self):
        return f"rooms_middle_visible_{self.GRID_SIZE}x{self.GRID_SIZE}"

    def get_gym_observation(self):
        minimap_obs = super().get_gym_observation()

        if self.selected_marine is not None:
            selected = [50, 205, 50]

            selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y
            local_x, local_y = self.global_to_local_pos(selected_marine_x, selected_marine_y)
            minimap_obs[:, local_y, local_x] = selected

        return minimap_obs


class SC2DefeatZerglingsAndBanelingsEnv(SC2GymWrapper):
    def __init__(self):
        super().__init__(screen_size=64, minimap_size=64)

        # move, attack, single select,
        # move = Function.ability(331, "Move_screen", cmd_screen, 3794), FUNCTIONS.Move_screen(3/queued [2]; 0/screen [84, 84])
        # attack = Function.ability(12, "Attack_screen", cmd_screen, 3674), FUNCTIONS.Attack_screen(3/queued [2]; 0/screen [84, 84])
        # single select = Function.ui_func(2, "select_point", select_point), FUNCTIONS.select_point(6/select_point_act [4]; 0/screen [84, 84])
        # multi select = Function.ui_func(3, "select_rect", select_rect), FUNCTIONS.select_rect(7/select_add [2]; 0/screen [84, 84]; 2/screen2 [84, 84])
        # select army = Function.ui_func(7, "select_army", select_army, lambda obs: obs.player_common.army_count > 0), FUNCTIONS.select_army(7/select_add [2])

        self.agent_actions = [
            FUNCTIONS.select_army,
            FUNCTIONS.select_point,
            FUNCTIONS.Move_screen,
            FUNCTIONS.Attack_screen,
        ]

        self.previous_units = list()

    @property
    def action_space(self):
        return spaces.MultiDiscrete([len(self.agent_actions), self.screen_size, self.screen_size])

    @property
    def observation_space(self):
        return spaces.Box(0, 255, (4, self.screen_size, self.screen_size), np.uint8)

    @property
    def name(self):
        return "defeat_zerg_bane"

    def init_sc2_env(self):
        sc2_env = SC2Env(
            map_name="DefeatZerglingsAndBanelings",
            players=[Agent(Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=self.screen_size, minimap=self.minimap_size),
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

        self.sc2_env = sc2_env
        self.obs = sc2_env.reset()[0]

    def get_gym_observation(self):
        feature_screen = self.obs.observation.feature_screen

        # features.PlayerRelative
        player_relative = feature_screen["player_relative"]
        unit_type = feature_screen["unit_type"]
        selected = feature_screen["selected"]
        unit_hit_points = feature_screen["unit_hit_points"]

        gym_obs = np.array([
            player_relative,
            unit_type,
            selected,
            unit_hit_points,
        ])

        return gym_obs

    def reward_function(self) -> int:
        reward = self.obs.reward

        # count banelings
        current_units = self.obs.observation.feature_units
        current_baneling_count = len([unit for unit in current_units if unit.unit_type == units.Zerg.Baneling])
        previous_baneling_count = len([unit for unit in self.previous_units if unit.unit_type == units.Zerg.Baneling])

        killed_banelings = previous_baneling_count - current_baneling_count
        if killed_banelings > 0:
            reward += killed_banelings * 5

        self.previous_units = current_units

        return reward

    def perform_action(self, action) -> None:
        action_type, screen_x, screen_y = action

        sc2_action = [FUNCTIONS.no_op()]
        agent_action = self.agent_actions[action_type]

        if agent_action.id not in self.obs.observation.available_actions:
            sc2_action = [FUNCTIONS.no_op()]

        elif action_type == 0:
            sc2_action = [agent_action("select")]

        elif action_type == 1:
            sc2_action = [agent_action("select", [screen_x, screen_y])]

        elif action_type in (2, 3):
            sc2_action = [agent_action("now", [screen_x, screen_y])]

        time_step = self.sc2_env.step(sc2_action)
        self.obs = time_step[0]

    def get_step_info(self) -> dict[str, Any]:
        info = dict()

        info["score"] = self.obs.observation['score_cumulative']['score']

        return info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.perform_action(action)

        obs = self.get_gym_observation()
        reward = self.reward_function()
        done = self.obs.last()
        truncated = False
        info = self.get_step_info()

        return obs, reward, done, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        return None

    def get_game_time(self):
        # Get the current game loop
        game_loop = self.obs.observation.game_loop[0]

        # Convert game loops to seconds (22.4 loops per second)
        seconds = game_loop / 22.4
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)

        return f"{minutes:02d}:{seconds:02d}"


class SC2BuildMarinesEnv(SC2GymWrapper):
    def __init__(self):
        screen_size, minimap_size = 64, 64
        super().__init__(screen_size, minimap_size)

        # select_point = Function.ui_func(2, "select_point", select_point),  FUNCTIONS.select_point(select_point_act [4]; screen [64, 64])
        # select_idle_worker = Function.ui_func(6, "select_idle_worker", select_idle_worker, lambda obs: obs.player_common.idle_worker_count > 0), FUNCTIONS.select_idle_worker(select_worker [4])
        # train_SCV = Function.ability(490, "Train_SCV_quick", cmd_quick, 524), FUNCTIONS.Train_SCV_quick(queued [2])
        # train_Marine = Function.ability(477, "Train_Marine_quick", cmd_quick, 560),  FUNCTIONS.Train_Marine_quick(queued [2])
        # build_SupplyDepot = Function.ability(91, "Build_SupplyDepot_screen", cmd_screen, 319), FUNCTIONS.Build_SupplyDepot_screen(queued [2], screen [64, 64])
        # build_Barracks = Function.ability(42, "Build_Barracks_screen", cmd_screen, 321), FUNCTIONS.Build_Barracks_screen(queued [2], screen [64, 64])

        self.agent_actions = [
            FUNCTIONS.select_point,
            FUNCTIONS.select_idle_worker,
            FUNCTIONS.Train_SCV_quick,
            FUNCTIONS.Train_Marine_quick,
            FUNCTIONS.Build_SupplyDepot_screen,
            FUNCTIONS.Build_Barracks_screen
        ]

    def init_sc2_env(self):
        sc2_env = SC2Env(
            map_name="BuildMarines",
            players=[Agent(Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=self.screen_size, minimap=self.minimap_size),
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

        self.sc2_env = sc2_env
        self.obs = sc2_env.reset()[0]

    @property
    def action_space(self):
        return spaces.MultiDiscrete([len(self.agent_actions), self.screen_size, self.screen_size])

    @property
    def observation_space(self):
        observation_space = spaces.Dict({
            "player": spaces.MultiDiscrete([np.iinfo(np.int32).max, 200 + 1, 200 + 1]),
            "screen": spaces.Box(0, 255, (5, self.screen_size, self.screen_size), np.uint8),
        })

        return observation_space

    @property
    def name(self):
        return f"build_marines"

    def get_gym_observation(self):
        player_minerals = self.obs.observation.player['minerals']
        player_food_used = self.obs.observation.player['food_used']
        player_food_cap = self.obs.observation.player['food_cap']
        player = np.array([player_minerals, player_food_used, player_food_cap])

        screen_player_id = self.obs.observation.feature_screen['player_id']
        screen_unit_type = self.obs.observation.feature_screen['unit_type']
        screen_selected = self.obs.observation.feature_screen['selected']
        screen_build_progress = self.obs.observation.feature_screen['build_progress']
        screen_pathable = self.obs.observation.feature_screen['pathable']

        mineral_field = 255
        screen_unit_type[screen_unit_type == units.Neutral.MineralField] = mineral_field
        screen_unit_type[screen_unit_type == units.Neutral.MineralField750] = mineral_field
        screen_unit_type[screen_unit_type == units.Neutral.MineralField450] = mineral_field

        screen = np.array([
            screen_player_id,
            screen_unit_type,
            screen_selected,
            screen_build_progress,
            screen_pathable
        ], dtype=np.uint8)

        gym_obs = dict()
        gym_obs["player"] = player
        gym_obs["screen"] = screen

        return gym_obs

    def perform_action(self, action):
        action_type, screen_x, screen_y = action

        sc2_action = [FUNCTIONS.no_op()]
        agent_action = self.agent_actions[action_type]

        if agent_action.id not in self.obs.observation.available_actions:
            sc2_action = [FUNCTIONS.no_op()]

        elif action_type == 0:
            sc2_action = [agent_action("select", [screen_x, screen_y])]

        elif action_type == 1:
            sc2_action = [agent_action("select")]

        elif action_type in (2, 3):
            sc2_action = [agent_action("now")]

        elif action_type in (4, 5):
            sc2_action = [agent_action("now", [screen_x, screen_y])]

        time_step = self.sc2_env.step(sc2_action)
        self.obs = time_step[0]

    def reward_function(self):
        reward = self.obs.reward

        return reward

    def get_step_info(self):
        info = dict()

        info["score"] = self.obs.observation['score_cumulative']['score']

        return info

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.perform_action(action)

        obs = self.get_gym_observation()
        reward = self.reward_function()
        done = self.obs.last()
        truncated = False
        info = self.get_step_info()

        return obs, reward, done, truncated, info

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        pass


if __name__ == "__main__":
    test_env = None

    try:
        test_env = SC2BuildMarinesEnv()
        check_env(test_env)

        # for _ in range(240):
        #     random_action = test_env.action_space.sample()
        #     test_env.step(random_action)
        #     # test_env.render()

    finally:
        if test_env is not None:
            test_env.close()
