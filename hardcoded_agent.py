import numpy as np
import pysc2.maps
from pysc2.agents import base_agent
from pysc2.env import sc2_env
from pysc2.lib import actions, features, units
from absl import app

_PLAYER_SELF = features.PlayerRelative.SELF
_PLAYER_NEUTRAL = features.PlayerRelative.NEUTRAL  # beacon/minerals
_PLAYER_ENEMY = features.PlayerRelative.ENEMY

FUNCTIONS = actions.FUNCTIONS
RAW_FUNCTIONS = actions.RAW_FUNCTIONS
TYPES = actions.TYPES
ABILITY_IDS = actions.ABILITY_IDS


def _xy_locs(mask):
    """Mask should be a set of bools from comparison with a feature layer."""
    y, x = mask.nonzero()
    return list(zip(x, y))


class HardcodedCollectMineralShards(base_agent.BaseAgent):

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


class HardcodedFindAndDefeatZerglings(base_agent.BaseAgent):

    def step(self, obs):

        observation = obs.observation

        for action in observation.available_actions:
            print(actions.FUNCTIONS[action])

        my_obs = np.zeros(shape=(2, 32, 32), dtype=np.int32)

        player_relative = observation.feature_minimap.player_relative
        unit_type = observation.feature_minimap.unit_type
        selected = observation.feature_minimap.selected
        visibility = observation.feature_minimap.visibility

        # player_relative = observation.feature_screen.player_relative
        # player_id = observation.feature_screen.player_id

        return FUNCTIONS.no_op()


def main(argv):
    agent = HardcodedFindAndDefeatZerglings()

    # put while cycle here for infinity games

    with sc2_env.SC2Env(
            map_name="FindAndDefeatZerglings",
            players=[sc2_env.Agent(sc2_env.Race.zerg)],
            agent_interface_format=features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=64, minimap=64),
                use_camera_position=True,
                use_feature_units=True),
            step_mul=8,
            realtime=True,
            visualize=True) as env:

        agent.setup(env.observation_spec(), env.action_spec())

        time_steps = env.reset()
        agent.reset()

        while True:
            step_actions = [agent.step(time_steps[0])]
            if time_steps[0].last():
                break
            time_steps = env.step(step_actions)


if __name__ == '__main__':
    app.run(main)

    # for map in pysc2.maps.get_maps():
    #     print(map)
