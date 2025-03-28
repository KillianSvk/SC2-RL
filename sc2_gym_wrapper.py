from absl import logging, flags
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.env.sc2_env import SC2Env, AgentInterfaceFormat, Agent

FLAGS = flags.FLAGS
FLAGS(["run.py"])

FUNCTIONS = actions.FUNCTIONS
_PLAYER = 1
_NEUTRAL = 3


class SC2GymEnvironment(gym.Env):
    """ Wraps PySC2's SC2Env to make it Gym-compatible. """

    def __init__(self):
        super(SC2GymEnvironment, self).__init__()

        self.reward = 0
        self.episodes = 0
        self.steps = 0

        self.screen_size = 32
        self.minimap_size = 32

        self.selected_marine = None
        self.GRID_SIZE = 7
        self.GRID_HALF_SIZE = self.GRID_SIZE // 2

        self.sc2_env = SC2Env(
            map_name="CollectMineralShards",
            players=[Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=self.screen_size, minimap=self.minimap_size),
                use_feature_units=True,
                crop_to_playable_area=True
            ),
            game_steps_per_episode=0,
            step_mul=8,
            realtime=False,
            visualize=False
        )

        self.obs = self.sc2_env.reset()[0]

        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()

    def _define_action_space(self):
        """Defines the Gym-compatible action space for PySC2."""

        actions_num = (self.GRID_SIZE * self.GRID_SIZE)

        # actions_num = (len(self.obs.observation.available_actions))  # Number of possible actions in PySC2

        return spaces.Discrete(actions_num)

    def _define_observation_space(self):
        """Defines the Gym-compatible observation space."""
        # observation_space = spaces.Box(low=0, high=4, shape=(self.screen_size, self.screen_size), dtype=np.int32)

        observation_space = spaces.Box(low=-1, high=1, shape=(self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)

        return observation_space

    def _xy_locs(self, mask):
        """Mask should be a set of bools from comparison with a feature layer."""
        y, x = mask.nonzero()
        return list(zip(x, y))

    def is_in_local_grid(self, x, y) -> bool:
        assert self.selected_marine is not None, "No marine selected yet, therefor no local grid"
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
        assert self.selected_marine is not None, "No marine selected yet, therefor no local grid"

        selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y
        local_x = x - (selected_marine_x - self.GRID_HALF_SIZE)
        local_y = y - (selected_marine_y - self.GRID_HALF_SIZE)

        return local_x, local_y

    def get_gym_observation(self):
        screen = self.obs.observation.feature_screen.player_relative
        local_grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int8)
        selected_units = [unit for unit in self.obs.observation.feature_units if unit.is_selected]

        if len(selected_units) > 0:
            self.selected_marine = selected_units[0]

        if self.selected_marine is not None:
            selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y

            # assign out of bounds to local grid
            for local_y in range(-self.GRID_HALF_SIZE, self.GRID_HALF_SIZE):
                for local_x in range(-self.GRID_HALF_SIZE, self.GRID_HALF_SIZE):
                    global_x, global_y = selected_marine_x + local_x, selected_marine_y + local_y
                    if not self.is_in_bounds(global_x, global_y):
                        local_grid[local_y + self.GRID_HALF_SIZE, local_x + self.GRID_HALF_SIZE] = -1

            # assign minerals to local grid
            for y in range(self.screen_size):
                for x in range(self.screen_size):
                    if screen[y][x] == _NEUTRAL and self.is_in_local_grid(x, y):
                        local_x, local_y = self.global_to_local_pos(x, y)
                        local_grid[local_y, local_x] = 1

        # print("-------------------------")
        # for line in local_grid.tolist():
        #     print(line)

        return local_grid

    def reset(self, seed=None, options=None):
        self.episodes += 1

        self.obs = self.sc2_env.reset()[0]
        gym_observation = self.get_gym_observation()
        info = {}

        return gym_observation, info

    def local_to_global_action(self, action) -> tuple[int, int]:
        if self.selected_marine is None:
            return 0, 0

        selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y
        local_x, local_y = action % self.GRID_SIZE, action // self.GRID_SIZE
        local_x, local_y = local_x - self.GRID_HALF_SIZE, local_y - self.GRID_HALF_SIZE
        global_x, global_y = local_x + selected_marine_x, local_y + selected_marine_y

        return global_x, global_y

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
            marines_pos = self._xy_locs(player_relative == _PLAYER)
            marine_pos = marines_pos[0]
            sc2_action = [FUNCTIONS.select_point("select", marine_pos)]

        time_step = self.sc2_env.step(sc2_action)

        # Unpack the returned values
        self.obs = time_step[0]
        score = self.obs.observation['score_cumulative']['score']
        reward = self.obs.reward
        done = self.obs.last()

        if done and score == 20:
            reward += 100

        self.reward += reward
        self.steps += 1

        # Truncated flag (e.g., if the episode ended due to time constraints)
        truncated = False

        # Info dictionary (optional debug info)
        info = {}

        return self.get_gym_observation(), reward, done, truncated, info

    def render(self, mode="human"):
        """ Optional: Implement visualization. """
        pass

    def close(self):
        """ Close the SC2 environment. """
        self.sc2_env.reset()
        self.sc2_env.close()
