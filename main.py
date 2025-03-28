import os
import time
from absl import flags, app

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

import torch
from torch.optim import Adam, RMSprop, AdamW

from sc2_gym_wrapper import SC2GymEnvironment
from gpu_logging import GPUTensorBoardCallback

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
FLAGS = flags.FLAGS
AGENTS_FOLDER = 'agents/'


def run_from_cmd(argv):
    rl_algorithm = None

    if argv[1] == 'dqn':
        rl_algorithm = DQN

    elif argv[1] == 'ppo':
        rl_algorithm = PPO

    if rl_algorithm is None:
        print("Wrong or None algorithm was chosen!")
        return

    model_path = argv[1]

    if argv[2] == 'train':
        train(rl_algorithm, model_path)

    elif argv[2] == 'test':
        test(rl_algorithm, model_path)


# Adam    learning_rate=1e-4
# RMSprop	learning_rate=2.5e-4

# model = rl_algorithm(
#        policy="MultiInputPolicy",
#        env=env,
#        verbose=1,
#        tensorboard_log="tensor_log",
#        gradient_steps=8,
#        buffer_size=50_000,
#        batch_size=128,
#        target_update_interval=1_000,
#        device="cuda"
#    )
#
#    model.learn(
#        total_timesteps=50_000,
#        progress_bar=True,
#    )

def make_env():
    time.sleep(1)
    return SC2GymEnvironment()


def train(rl_algorithm, model_path):
    env = None
    num_envs = 5

    try:
        # env = SC2GymEnvironment()
        env = SubprocVecEnv([lambda: make_env() for _ in range(num_envs)])

        model = rl_algorithm(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            gradient_steps=8,
            buffer_size=1_000_000,
            batch_size=128,
            target_update_interval=1_000,
            exploration_fraction=0.2,
            device="cuda"
        )

        model.learn(
            total_timesteps=100_000,
            callback=GPUTensorBoardCallback(),
            progress_bar=True
        )

        model.save(AGENTS_FOLDER + model_path)

    finally:
        if env is not None:
            env.close()


def test(rl_algorithm, model_path):
    env = SC2GymEnvironment()

    try:
        model = rl_algorithm.load(
            path=AGENTS_FOLDER + model_path,
            env=env,
            device="cuda"
        )

        assert model is not None, "Model failed to load!"

        obs, info = env.reset()
        total_reward = 0
        episode_reward = 0
        episodes = 0
        actions_dict = dict()

        while True:
            action, _states = model.predict(obs)
            action_int = int(action)

            if action_int not in actions_dict.keys():
                actions_dict[action_int] = 0

            actions_dict[action_int] += 1

            obs, reward, done, truncated, info = env.step(action)

            episode_reward += reward
            total_reward += reward

            if done or truncated:
                episodes += 1

                print(actions_dict)

                print(f"Episode reward: {episode_reward}")
                print(f"Average episode reward: {total_reward / episodes:.2f} after {episodes} episodes")
                print("-------------------------------------")

                obs, info = env.reset()
                episode_reward = 0
                actions_dict = dict()

    finally:
        env.close()


def main(argv):
    if len(argv) > 1:
        run_from_cmd(argv)
        return

    # IF NOT CMD
    rl_algorithm = DQN
    model_path = 'dqn'

    train(rl_algorithm, model_path)
    # test(rl_algorithm, model_path)


# scp -r C:\Users\petoh\Desktop\School\Bakalarka\web\index.html hozlar5@davinci.fmph.uniba.sk:~/public_html/bakalarska_praca/
# tensorboard --logdir=tensor_log
if __name__ == "__main__":
    app.run(main)