import os
import time
from absl import flags, app
import psutil

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.logger import configure

import torch
from torch.optim import Adam, RMSprop, AdamW

from sc2_gym_wrapper import *
from agent_logging import CustomMetricsCallback

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
FLAGS = flags.FLAGS

AGENTS_FOLDER = 'agents'
ENV = SC2GymEnvironment
ALGORITHM = DQN

def run_from_cmd(argv):
    rl_algorithm = None

    if argv[1] == 'dqn':
        rl_algorithm = DQN

    elif argv[1] == 'ppo':
        rl_algorithm = PPO

    if rl_algorithm is None:
        print("Wrong or None algorithm was chosen!")
        return

    if argv[2] == 'train':
        train(rl_algorithm)

    elif argv[2] == 'test':
        test(rl_algorithm)


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
    return ENV()


def train(rl_algorithm):
    env = None
    num_envs = 6

    try:
        # env = make_env()
        # env_name = env.name

        env = SubprocVecEnv([lambda: make_env() for _ in range(num_envs)])
        env_names = env.get_attr("name")
        env_name = env_names[0]

        model = rl_algorithm(
            policy="MlpPolicy",
            tensorboard_log="tensorboard",
            env=env,
            verbose=1,
            gradient_steps=8,
            buffer_size=1_000_000,
            batch_size=128,
            target_update_interval=1_000,
            exploration_fraction=0.2,
            device="cuda"
        )

        model_name = model.__class__.__name__
        agent_name = model_name + "_" + env_name

        agent_folder_name = agent_name + "_" + time.strftime("%d-%m_%H-%M-%S")
        log_path = os.path.join("tensorboard", agent_folder_name)
        new_logger = configure(log_path, ["stdout", "tensorboard"])
        model.set_logger(new_logger)

        TIMESTEPS = 50_000
        for i in range(5):
            model.learn(
                total_timesteps=TIMESTEPS,
                callback=CustomMetricsCallback(),
                log_interval=4,
                tb_log_name="",
                progress_bar=True,
                reset_num_timesteps=False
                )

            model_path = os.path.join(AGENTS_FOLDER, agent_folder_name, agent_name + "_" + f"{(i+1)*TIMESTEPS//1_000}k")
            model.save(model_path)
        # model.save(AGENTS_FOLDER + "/" + model_name + "_" + f"{TIMESTEPS//1_000}k")

    finally:
        if env is not None:
            env.close()

        parent = psutil.Process(os.getpid())
        for child in parent.children(recursive=True):
            child.terminate()  # Gracefully terminate
        gone, alive = psutil.wait_procs(parent.children(), timeout=3)  # Wait for cleanup

        # If any processes are still alive, force kill them
        for child in alive:
            child.kill()


def get_latest_model_path():
    existing_runs = sorted(
        os.listdir(AGENTS_FOLDER),
        key=lambda x: os.path.getctime(os.path.join(AGENTS_FOLDER, x)),
        reverse=True
    )
    last_run = existing_runs[0]
    last_run = last_run.split(".")[0]
    return os.path.join(AGENTS_FOLDER, last_run)


def test(rl_algorithm):
    env = make_env()
    # model_path = get_latest_model_path()
    model_path = "agents/local_grid_env_11x11/local_grid_env_11x11_1250k"

    try:
        model = rl_algorithm.load(
            path=model_path,
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

    train(ALGORITHM)
    # test(ALGORITHM)


# scp -r C:\Users\petoh\Desktop\School\Bakalarka\web\index.html hozlar5@davinci.fmph.uniba.sk:~/public_html/bakalarska_praca/
# tensorboard --logdir=tensorboard
if __name__ == "__main__":
    app.run(main)
