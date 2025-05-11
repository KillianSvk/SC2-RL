import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import DQN, PPO

from utils import get_latest_model_path
from sc2_environments import *


ENV = SC2DefeatZerglingsAndBanelingsEnv
ALGORITHM = PPO
NUM_TESTING_EPISODES = 10
MODEL_PATH = get_latest_model_path()
# MODEL_PATH = "agents/DQN_screen_36x36_22-04_00-15/DQN_screen_36x36_45000k"


def test(algorithm):
    env = ENV()

    model = algorithm.load(
        path=MODEL_PATH,
        env=env,
        device="cuda"
    )

    assert model is not None, "Model failed to load!"

    obs, info = env.reset()
    total_reward = 0
    total_score = 0
    episode_reward = 0
    best_score = 0
    episodes = 0

    while episodes < NUM_TESTING_EPISODES:
        action, _states = model.predict(obs)

        obs, reward, done, truncated, info = env.step(action)
        env.render()

        episode_reward += reward
        total_reward += reward

        if done or truncated:
            episodes += 1

            episode_score = info["score"]
            total_score += episode_score

            if episode_score > best_score:
                best_score = episode_score

            print(f"Episode reward: {episode_reward} Episode score: {episode_score}")
            print(f"Average episode reward: {total_reward / episodes:.2f}")
            print(f"Average episode score: {total_score / episodes:.2f}")
            print(f"Best score: {best_score} out of {episodes} episodes")
            print("-------------------------------------")

            obs, info = env.reset()
            episode_reward = 0

    env.close()


if __name__ == '__main__':
    test(ALGORITHM)
