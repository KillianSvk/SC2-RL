import os
import time

import pandas as pd

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from utils import get_latest_model_path, get_latest_model_checkpoint, TEST_RESULTS_FOLDER, make_vec_env, AGENTS_FOLDER
from sc2_environments import *


ENV = SC2DefeatZerglingsAndBanelingsEnv
ALGORITHM = PPO
NUM_ENVS = 1
NUM_TESTING_EPISODES = 100
# MODEL_PATH = get_latest_model_path()
MODEL_PATH = "agents/PPO_defeat_zerg_bane_18-05_00-19/PPO_defeat_zerg_bane_10002k.zip"

# MODEL_NAMES = [
#     "A2C_screen_36x36_11-05_23-22",
#     "A2C_screen_36x36_16-05_23-42",
#     "A2C_screen_box_36x36_19-05_00-24",
# ]
#
# MODEL_CHECKPOINTS = [
#     "A2C_screen_36x36_10000k.zip",
#     "A2C_screen_36x36_50000k.zip",
#     "A2C_screen_36x36_90000k.zip",
# ]

PRINT_RESULTS = False
SAVE_RESULTS = True


def test(algorithm):
    model_start_time = time.strftime('%d-%m_%H-%M')
    env = make_vec_env(ENV, NUM_ENVS)
    # env = ENV()

    env_name = env.get_attr("name")[0]

    # model_path = os.path.join(AGENTS_FOLDER, model_name, model_checkpoint_name)
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
        # path = os.path.join(TEST_RESULTS_FOLDER, f"{agent_name}_test.csv")
        path = os.path.join(TEST_RESULTS_FOLDER, f"test_{model_start_time}.csv")
        df.to_csv(path, index=False)

        print("results saved")


def test_random_agent():
    model_start_time = time.strftime('%d-%m_%H-%M')
    env = make_vec_env(ENV, NUM_ENVS)

    env_name = env.get_attr("name")[0]

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
        action = [env.action_space.sample() for _ in range(NUM_ENVS)]
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
        path = os.path.join(TEST_RESULTS_FOLDER, f"{env_name}_test_{model_start_time}.csv")
        df.to_csv(path, index=False)

        print("results saved")


if __name__ == '__main__':
    test(ALGORITHM)
    # test_random_agent()
    # for model in MODEL_NAMES:
    #     print(model, get_latest_model_checkpoint(model))

    # agent_paths = [
    #     "DQN_screen_36x36_03-06_05-15",
    #     # "DQN_screen_36x36_11-05_00-52",
    #     # "DQN_screen_36x36_30-05_23-51",
    # ]

    # agent_paths = [
    #     # ("PPO_screen_36x36_12-05_09-38", "PPO_screen_36x36_10002k.zip"),
    #     ("PPO_screen_box_36x36_03-06_08-52", "PPO_screen_box_36x36_10014k.zip"),
    #     ("PPO_screen_box_36x36_31-05_23-54", "PPO_screen_box_36x36_10002k.zip"),
    # ]
    #
    # for i, (agent_path, checkpoint) in enumerate(agent_paths):
    #     MODEL_PATH = os.path.join(AGENTS_FOLDER, agent_path, checkpoint)
    #     print(f"testing: {agent_path}")
    #     test(agent_path, ALGORITHM)
