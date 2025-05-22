import os
import time

import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from utils import get_latest_model_path, TEST_RESULTS_FOLDER, make_envs
from sc2_environments import *


ENV = SC2ScreenEnv
ALGORITHM = A2C
NUM_ENVS = 3
NUM_TESTING_EPISODES = 100
# MODEL_PATH = get_latest_model_path()
MODEL_PATH = "agents/A2C_screen_36x36_16-05_23-42/A2C_screen_36x36_10000k.zip"
PRINT_RESULTS = True
SAVE_RESULTS = True


def test(algorithm):
    start_time = time.strftime('%d-%m_%H-%M')
    env = make_envs(ENV, NUM_ENVS)

    model = algorithm.load(
        path=MODEL_PATH,
        env=env,
        device="cuda"
    )

    assert model is not None, "Model failed to load!"

    obs = env.reset()
    total_reward = 0
    total_score = 0
    best_score = 0
    episodes = 0
    episode_rewards = 0

    all_scores = []
    all_rewards = []

    while episodes < NUM_TESTING_EPISODES:
        action, _states = model.predict(obs)

        obs, rewards, dones, infos = env.step(action)

        episode_rewards += sum(rewards)
        total_reward += sum(rewards)

        if any(dones):
            episodes += 1 * NUM_ENVS

            episode_scores = sum([info['score'] for info in infos])
            total_score += episode_scores

            all_scores.append(episode_scores)
            all_rewards.append(episode_rewards)

            if episode_scores > best_score:
                best_score = episode_scores

            if PRINT_RESULTS:
                print(f"Episode: {episodes}")
                print(f"Episode rewards: {episode_rewards}")
                print(f"Episode score: {episode_scores}")
                print()
                print(f"Average episode reward: {total_reward / episodes:.2f}")
                print(f"Average episode score: {total_score / episodes:.2f}")
                print(f"Best score: {best_score}")
                print("-------------------------------------")

            obs = env.reset()
            episode_rewards = 0

    env.close()

    if SAVE_RESULTS:
        all_scores = np.array(all_scores)
        all_rewards = np.array(all_rewards)

        df = pd.DataFrame({
            'episode': np.arange(1, NUM_TESTING_EPISODES + 1),
            'reward': all_rewards,
            'score': all_scores
        })

        os.makedirs(TEST_RESULTS_FOLDER, exist_ok=True)
        path = os.path.join(TEST_RESULTS_FOLDER, f"test_{start_time}.csv")
        df.to_csv(path, index=False)

        print("results saved")


if __name__ == '__main__':
    test(ALGORITHM)
