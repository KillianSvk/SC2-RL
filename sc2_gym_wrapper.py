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

    def __init__(self, realtime=False):
        super(SC2GymEnvironment, self).__init__()

        self.reward = 0
        self.episodes = 0
        self.steps = 0

        self.screen_size = 32
        self.minimap_size = 32

        self.selected_marine_pos = None

        self.sc2_env = SC2Env(
            map_name="CollectMineralShards",
            players=[Agent(sc2_env.Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=self.screen_size, minimap=self.minimap_size),
                use_feature_units=True,
                use_raw_units=True
            ),
            game_steps_per_episode=0,
            step_mul=8,
            realtime=realtime,
            visualize=False
        )

        self.obs = self.sc2_env.reset()[0]

        self.action_space = self._define_action_space()
        self.observation_space = self._define_observation_space()

    def _define_action_space(self):
        """Defines the Gym-compatible action space for PySC2."""

        actions_num = (self.screen_size * self.screen_size)

        # actions_num = (len(self.obs.observation.available_actions))  # Number of possible actions in PySC2

        return spaces.Discrete(actions_num)

    def _define_observation_space(self):
        """Defines the Gym-compatible observation space."""
        # observation_space = spaces.Box(low=0, high=4, shape=(self.screen_size, self.screen_size), dtype=np.int32)

        observation_space = spaces.Dict({
            "position": spaces.Box(low=0, high=1, shape=(self.screen_size, self.screen_size), dtype=np.int32),
            "minerals": spaces.Box(low=0, high=1, shape=(self.screen_size, self.screen_size), dtype=np.int32),
        })

        return observation_space

    def _xy_locs(self, mask):
        """Mask should be a set of bools from comparison with a feature layer."""
        y, x = mask.nonzero()
        return list(zip(x, y))

    def get_gym_observation(self):
        minerals_screen = np.array([[0 for x in range(self.screen_size)] for y in range(self.screen_size)])
        position_screen = np.array([[0 for x in range(self.screen_size)] for y in range(self.screen_size)])
        screen = self.obs.observation.feature_screen.player_relative

        for y in range(self.screen_size):
            for x in range(self.screen_size):
                if screen[y][x] == _NEUTRAL:
                    minerals_screen[y][x] = 1

        if self.selected_marine_pos is not None:
            selected_marine_x, selected_marine_y = self.selected_marine_pos
            position_screen[selected_marine_y // self.screen_size][selected_marine_x // self.screen_size] = 1

        observation_space = dict()
        observation_space["minerals"] = minerals_screen
        observation_space["position"] = position_screen

        return observation_space

    def reset(self, seed=None, options=None):
        self.episodes += 1

        self.obs = self.sc2_env.reset()[0]
        gym_observation = self.get_gym_observation()
        info = {}

        return gym_observation, info

    def step(self, action):
        """Executes the given action in the SC2 environment and returns the new state, reward, and episode status."""

        # function_id = int(action)  # Convert Gym action to PySC2 function ID
        # available_actions = self.sc2_env.observation_spec()[0]['available_actions']
        #
        # if function_id not in available_actions:
        #     # print(f"Action id: {function_id} is not available.")
        #     function_id = actions.FUNCTIONS.no_op.id  # Default to no-op
        #
        # # Example of a simple action with no parameters
        # action_call = actions.FunctionCall(function_id, [])
        #
        # # Apply the action and advance the environment
        # time_step = self.sc2_env.step([action_call])

        x = action % self.screen_size
        y = action // self.screen_size

        player_relative = self.obs.observation.feature_screen.player_relative
        available_actions = self.obs.observation.available_actions

        if FUNCTIONS.Move_screen.id in available_actions:
            sc2_action = [FUNCTIONS.Move_screen("now", (x, y))]

        else:
            # sc2_action = [FUNCTIONS.select_army("select")]
            marines_pos = self._xy_locs(player_relative == _PLAYER)
            marine_pos = marines_pos[0]
            self.selected_marine_pos = marine_pos
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
