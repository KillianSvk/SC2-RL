import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import DQN, PPO

from utils import get_latest_model_path
from sc2_environments import *


ENV = SC2DefeatZerglingsAndBanelingsEnv
ALGORITHM = PPO
NUM_TESTING_EPISODES = 100


def test(algorithm):
    env = ENV()

    model_path = get_latest_model_path()
    # model_path = "agents/DQN_screen_36x36_22-04_00-15/DQN_screen_36x36_45000k"

    model = algorithm.load(
        path=model_path,
        env=env,
        device="cuda"
    )

    assert model is not None, "Model failed to load!"

    obs, info = env.reset()
    total_reward = 0
    episode_reward = 0
    best_score = 0
    episodes = 0
    actions_dict = dict()

    while episodes < NUM_TESTING_EPISODES:
        action, _states = model.predict(obs)
        action_int = int(action)

        if action_int not in actions_dict.keys():
            actions_dict[action_int] = 0

        actions_dict[action_int] += 1

        obs, reward, done, truncated, info = env.step(action)
        env.render()

        episode_reward += reward
        total_reward += reward

        if done or truncated:
            episodes += 1

            # print(actions_dict)
            episode_score = info["score"]
            if episode_score > best_score:
                best_score = episode_score


            print(f"Episode reward: {episode_reward} Episode score: {episode_score}")
            print(f"Average episode reward: {total_reward / episodes:.2f} after {episodes} episodes")
            print(f"Best score: {best_score}")
            print("-------------------------------------")

            obs, info = env.reset()
            episode_reward = 0
            actions_dict = dict()

    env.close()


if __name__ == '__main__':
    test(ALGORITHM)
