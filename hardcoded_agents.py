import numpy as np
import pysc2.maps
from pysc2.agents.base_agent import BaseAgent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from pysc2.lib.actions import FUNCTIONS, FUNCTIONS_AVAILABLE, FUNCTION_TYPES
from pysc2.env.enums import Race
from pysc2.env.sc2_env import SC2Env, Agent
from absl import app

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

RAW_FUNCTIONS = actions.RAW_FUNCTIONS
TYPES = actions.TYPES
ABILITY_IDS = actions.ABILITY_IDS


def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))


class HardcodedCollectMineralShards(BaseAgent):

    def step(self, obs):
        if FUNCTIONS.Move_screen.id in obs.observation.available_actions:
            player_relative = obs.observation.feature_screen.player_relative
            minerals = _xy_locs(player_relative == _PLAYER_NEUTRAL)
            if not minerals:
                return FUNCTIONS.no_op()
            marines = _xy_locs(player_relative == _PLAYER_SELF)
            marine_xy = np.mean(marines, axis=0).round()  # Average location.
            distances = np.linalg.norm(np.array(minerals) - marine_xy, axis=1)
            closest_mineral_xy = minerals[np.argmin(distances)]
            return FUNCTIONS.Move_screen("now", closest_mineral_xy)

        return FUNCTIONS.select_army("select")


class HardcodedFindAndDefeatZerglings(BaseAgent):

    def step(self, obs):
        observation = obs.observation

        # for action in observation.available_actions:
        #     print(actions.FUNCTIONS[action])

        my_obs = np.zeros(shape=(2, 32, 32), dtype=np.int32)

        selected = observation.feature_screen.selected
        player_relative = observation.feature_screen.player_relative
        pathable = observation.feature_screen.pathable

        for line in list(pathable):
            print(line)

        # player_relative = observation.feature_screen.player_relative
        # player_id = observation.feature_screen.player_id

        return FUNCTIONS.no_op()


class RandomAgent(BaseAgent):
    def step(self, obs):
        super(RandomAgent, self).step(obs)

        function_id = np.random.choice(obs.observation.available_actions)
        args = [[np.random.randint(0, size) for size in arg.sizes]
                for arg in self.action_spec.functions[function_id].args]

        return actions.FunctionCall(function_id, args)


class TestAgent(BaseAgent):
    def __init__(self):
        super().__init__()

    def step(self, obs):
        available_actions = obs.observation.available_actions
        # print(available_actions)

        production_queue = obs.observation['production_queue']
        screen_active = obs.observation.feature_screen['active']

        return FUNCTIONS.no_op()


def main(argv):
    # agent = TestAgent()
    #
    # with sc2_env.SC2Env(
    #         map_name="BuildMarines",
    #         players=[sc2_env.Agent(sc2_env.Race.zerg)],
    #         agent_interface_format=features.AgentInterfaceFormat(
    #             feature_dimensions=features.Dimensions(screen=32, minimap=32),
    #             use_camera_position=True,
    #             use_feature_units=True),
    #         step_mul=8,
    #         realtime=True,
    #         visualize=True) as env:
    #
    #     agent.setup(env.observation_spec()[0], env.action_spec()[0])
    #
    #     time_steps = env.reset()
    #     agent.reset()
    #
    #     while True:
    #         step_actions = [agent.step(time_steps[0])]
    #         if time_steps[0].last():
    #             break
    #         time_steps = env.step(step_actions)

    with SC2Env(
            map_name="CollectMineralShards",
            players=[Agent(Race.terran)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=64, minimap=64),
                use_feature_units=True,
                use_camera_position=True,
                crop_to_playable_area=True,
                action_space=actions.ActionSpace.FEATURES,
            ),
            game_steps_per_episode=0,
            step_mul=8,
            realtime=True,
            visualize=False
    ) as env:
        time_steps = env.reset()

        while True:
            if time_steps[0].last():
                break
            time_steps = env.step([FUNCTIONS.no_op()])


if __name__ == '__main__':
    app.run(main)

    # for map in pysc2.maps.get_maps():
    #     print(map)
