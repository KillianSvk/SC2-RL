import os
import time

import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from utils import get_latest_model_path, TEST_RESULTS_FOLDER, make_vec_env, AGENTS_FOLDER
from sc2_environments import *


ENV = SC2ScreenEnv
ALGORITHM = DQN
NUM_ENVS = 6
NUM_TESTING_EPISODES = 100
# MODEL_PATH = get_latest_model_path()
MODEL_PATH = "agents/DQN_local_grid_11x11_23-05_03-35/DQN_local_grid_11x11_1000k.zip"
PRINT_RESULTS = False
SAVE_RESULTS = True


def test(algorithm):
    start_time = time.strftime('%d-%m_%H-%M')
    env = make_vec_env(ENV, NUM_ENVS)
    # env = ENV()

    env_name = env.get_attr("name")[0]

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

    envs_episode_rewards = dict()
    for env_id in range(NUM_ENVS):
        envs_episode_rewards[env_id] = []

    all_episodes_rewards = []
    all_episodes_scores = []

    while episodes < NUM_TESTING_EPISODES:
        action, _states = model.predict(obs)

        obs, rewards, dones, infos = env.step(action)

        for env_id, reward in enumerate(rewards):
            envs_episode_rewards[env_id].append(reward)

        for env_id, done in enumerate(dones):
            if done and episodes < NUM_TESTING_EPISODES:
                episodes += 1

                episode_score = infos[env_id]['score']
                total_score += episode_score
                all_episodes_scores.append(episode_score)

                episode_reward = sum(envs_episode_rewards[env_id])
                total_reward += episode_reward
                all_episodes_rewards.append(episode_reward)

                if episode_score > best_score:
                    best_score = episode_score

                if PRINT_RESULTS:
                    print(f"Episode: {episodes}")
                    print(f"Episode reward: {episode_reward}")
                    print(f"Episode score: {episode_score}")
                    print()
                    print(f"Average episode reward: {total_reward / episodes:.2f}")
                    print(f"Average episode score: {total_score / episodes:.2f}")
                    print(f"Best score: {best_score}")
                    print("-------------------------------------")

                obs = env.reset()
                envs_episode_rewards[env_id] = []

    env.close()

    if SAVE_RESULTS:
        all_episodes_scores = np.array(all_episodes_scores)
        all_episodes_rewards = np.array(all_episodes_rewards)

        df = pd.DataFrame({
            'episode': np.arange(1, NUM_TESTING_EPISODES + 1),
            'reward': all_episodes_rewards,
            'score': all_episodes_scores
        })

        os.makedirs(TEST_RESULTS_FOLDER, exist_ok=True)
        path = os.path.join(TEST_RESULTS_FOLDER, f"{env_name}_test_{start_time}.csv")
        df.to_csv(path, index=False)

        print("results saved")


if __name__ == '__main__':
    # test(ALGORITHM)

    agent_paths = [
        "DQN_screen_36x36_31-05_13-30",
        "DQN_screen_36x36_31-05_12-56",
        "DQN_screen_36x36_31-05_12-22",
        "DQN_screen_36x36_29-05_00-05",
        "DQN_screen_36x36_28-05_23-35",
        "DQN_screen_36x36_28-05_23-05",
    ]

    for agent_path in agent_paths:
        MODEL_PATH = os.path.join(AGENTS_FOLDER, agent_path, "DQN_screen_36x36_1000k.zip")
        test(ALGORITHM)
