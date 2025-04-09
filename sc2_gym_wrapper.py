from absl import logging, flags
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3.common.env_checker import check_env

from pysc2.env import sc2_env
from pysc2.lib import actions, features, units, portspicker
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

        self.episodes = 0
        self.steps = 0

        self.screen_size = 32
        self.minimap_size = 32

        self.selected_marine = None
        self.GRID_SIZE = 11
        self.GRID_HALF_SIZE = self.GRID_SIZE // 2

        self.name = f"dqn_local_grid_{self.GRID_SIZE}x{self.GRID_SIZE}"

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

        self.define_action_space()
        self.define_observation_space()

    def __str__(self):
        return f"{self.name}_{self.GRID_SIZE}x{self.GRID_SIZE}"

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

        # print("-------------------------")
        # for line in local_grid.tolist():
        #     print(line)

        # local_grid = local_grid.flatten().astype(np.int8)

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
        score = self.obs.observation['score_cumulative']['score']
        reward = self.obs.reward
        reward -= 0.01

        return reward

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
        info = {}

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

        self.name = "dqn_flatten_obs_env"
        self.GRID_SIZE = 11
        self.GRID_HALF_SIZE = self.GRID_SIZE // 2

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


class SC2BoxEnv(SC2GymEnvironment):
    def __init__(self):
        super(SC2BoxEnv, self).__init__()

        self.name = "box_obs_env"
        self.GRID_SIZE = 5
        self.GRID_HALF_SIZE = self.GRID_SIZE // 2

        self.define_action_space()
        self.define_observation_space()

    def define_action_space(self):
        self.action_space = spaces.Discrete(self.GRID_SIZE * self.GRID_SIZE)

    def define_observation_space(self):
        self.observation_space = spaces.Box(low=0, high=self.screen_size, shape=(20+1, 2), dtype=np.uint8)

    def get_gym_observation(self):
        gym_obs = np.zeros((21, 2), dtype=np.uint8)
        screen = self.obs.observation.feature_screen.player_relative
        selected_units = [unit for unit in self.obs.observation.feature_units if unit.is_selected]

        if len(selected_units) > 0:
            self.selected_marine = selected_units[0]

        if self.selected_marine is not None:
            selected_marine_x, selected_marine_y = self.selected_marine.x, self.selected_marine.y
            gym_obs[0][0], gym_obs[0][1] = selected_marine_x, selected_marine_y

        minerals = self.xy_locations(screen == _NEUTRAL)
        for i, mineral in enumerate(minerals, 1):
            gym_obs[i][0], gym_obs[i][1] = mineral

        # for line in gym_obs.tolist():
        #     print(line)
        # print()

        return gym_obs


if __name__ == "__main__":
    env = None

    try:
        env = SC2FlattenEnv()
        check_env(env)

    finally:
        if env is not None:
            env.close()
