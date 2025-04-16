import os
import time
from absl import flags, app
import psutil
import subprocess

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.env_util import make_vec_env

import torch
from torch.optim import Adam, RMSprop, AdamW

from sc2_gym_wrapper import *
from custom_features import *
from agent_logging import CustomCallback

FLAGS = flags.FLAGS

AGENTS_FOLDER = 'agents'
MONITOR_FOLDER = "monitor"

NUM_ENVS = 6
ENV = SC2GymEnvironment
ALGORITHM = DQN
POLICY = "MlpPolicy"
POLICY_KWARGS = dict(
    # features_extractor_class=CustomizableCNN,
    # features_extractor_kwargs=dict(features_dim=256),
    # normalize_images=False,

    # net_arch=[256, 256, 128]
    # activation_fn=nn.ReLU
)
TIMESTEPS = 50_000
SAVING_FREQ = 10_000


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


def get_env():
    return ENV()


def make_monitored_env(start_time=None, env_id=0):
    env = ENV()
    filename = os.path.join(MONITOR_FOLDER, f"{env.name}_{start_time}", f"{env_id}")
    monitored_env = Monitor(env, filename=filename)
    return monitored_env


def env_error_cleanup():
    # Terminate child processes of the current script
    parent = psutil.Process(os.getpid())
    children = parent.children(recursive=True)
    for child in children:
        child.terminate()

    gone, alive = psutil.wait_procs(children, timeout=3)
    for child in alive:
        child.kill()

    # Kill leftover SC2/Blizzard processes
    targets = ['sc2', 'starcraft', 'blizzard', 'blizzarderror']
    for proc in psutil.process_iter(['pid', 'name']):
        name = proc.info['name']
        if name and any(t in name.lower() for t in targets):
            proc.kill()


def make_envs(start_time):
    env = None

    # Fixes weird Sc2 broken pipe error during init
    while env is None:
        try:
            # env = make_vec_env()
            env = SubprocVecEnv([lambda i=i: make_monitored_env(start_time, i) for i in range(NUM_ENVS)])

        except BrokenPipeError as error:
            env_error_cleanup()

    return env


def train(rl_algorithm):
    start_time = time.strftime('%d-%m_%H-%M')
    env = make_envs(start_time)

    env_names = env.get_attr("name")
    env_name = env_names[0]

    model = rl_algorithm(
        env=env,
        policy=POLICY,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log="tensorboard",
        device="cuda",
    )

    model_name = model.__class__.__name__
    agent_name = model_name + "_" + env_name
    agent_folder_name = agent_name + "_" + start_time

    callback = CustomCallback(
        save_freq=SAVING_FREQ,
        save_path=os.path.join(AGENTS_FOLDER, agent_folder_name),
        model_name=agent_name
    )

    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callback,
        log_interval=4,
        tb_log_name=agent_folder_name,
        progress_bar=True,
        reset_num_timesteps=False
    )

    env.close()


def get_latest_model_path():
    existing_agents = sorted(
        os.listdir(AGENTS_FOLDER),
        key=lambda x: os.path.getctime(os.path.join(AGENTS_FOLDER, x)),
        reverse=True
    )

    last_agent = existing_agents[0]
    agent_path = os.path.join(AGENTS_FOLDER, last_agent)
    agent_models = sorted(
        os.listdir(agent_path),
        key=lambda x: os.path.getctime(os.path.join(agent_path, x)),
        reverse=True
    )

    last_agent_model = agent_models[0]
    last_agent_model = last_agent_model.split(".")[0]

    return os.path.join(AGENTS_FOLDER, last_agent, last_agent_model)


def test(rl_algorithm):
    env = get_env()
    model_path = get_latest_model_path()
    # model_path = "agents/DQN_screen_36x36_12-04_17-03-47/DQN_screen_36x36_300k"
    num_testing_episodes = 20

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

        while episodes < num_testing_episodes:
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

    except KeyboardInterrupt:
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