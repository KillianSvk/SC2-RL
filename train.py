import os
import time

from stable_baselines3 import DQN, PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env

from utils import AGENTS_FOLDER
from utils import make_envs
from sc2_environments import *
from agent_logging import CustomCallback

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


NUM_ENVS = 6
ENV = SC2ScreenEnv
ALGORITHM = DQN
POLICY = "CnnPolicy" #MlpPolicy/CnnPolicy
POLICY_KWARGS = dict(
    # features_extractor_class=CustomizableCNN,
    # features_extractor_kwargs=dict(features_dim=256),
    # normalize_images=False,

    # net_arch=[256, 256, 128]
    # activation_fn=nn.ReLU
)
TIMESTEPS = 50_000
SAVING_FREQ = 10_000


def train(algorithm):
    start_time = time.strftime('%d-%m_%H-%M')
    env = make_envs(ENV, NUM_ENVS)
    # env = make_envs(ENV, NUM_ENVS, start_time)

    env_names = env.get_attr("name")
    env_name = env_names[0]

    # model = algorithm(
    #     env=env,
    #     policy=POLICY,
    #     policy_kwargs=POLICY_KWARGS,
    #     tensorboard_log="tensorboard",
    #     device="cuda",
    # )
    #

    model_path = "agents/DQN_screen_36x36_20-04_00-43/DQN_screen_36x36_30000k.zip"
    model = algorithm.load(
        path=model_path,
        env=env,
        device="cuda"
    )

    model_name = model.__class__.__name__
    agent_name = model_name + "_" + env_name
    agent_folder_name = agent_name + "_" + start_time


    # model_path_arr = model_path.split("/")
    # agent_folder_name = model_path_arr[1]
    # agent_folder_name_arr = agent_folder_name.split("_")
    # agent_name = "_".join(agent_folder_name_arr[:-2])

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


if __name__ == '__main__':
    train(ALGORITHM)
