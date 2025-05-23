import os
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import A2C, DDPG, DQN, PPO, SAC, TD3
from utils import AGENTS_FOLDER, make_envs, get_latest_model_path
from sc2_environments import *
from agent_logging import CustomCheckpointCallback


ENV = SC2LocalObservationEnv
NUM_ENVS = 6
ALGORITHM = DQN
POLICY = "MlpPolicy" #MlpPolicy/CnnPolicy/MultiInputPolicy
POLICY_KWARGS = dict(
    # features_extractor_class=CustomizableCNN,
    # features_extractor_kwargs=dict(features_dim=256),
    # normalize_images=False,
    # net_arch=[256, 256, 128]
    # activation_fn=nn.ReLU
)
TIMESTEPS = 1_000_000
SAVING_FREQ = 250_000

CONTINUE_MODEL_PATH = get_latest_model_path()
# CONTINUE_MODEL_PATH = "agents/DQN_middle_invisible_48x48_26-04_00-17/DQN_middle_invisible_48x48_15000k"


def train(algorithm):
    start_time = time.strftime('%d-%m_%H-%M')

    env = make_envs(ENV, NUM_ENVS)

    env_names = env.get_attr("name")
    env_name = env_names[0]

    model = algorithm(
        env=env,
        policy=POLICY,
        policy_kwargs=POLICY_KWARGS,
        tensorboard_log="tensorboard_collect_minerals",
    )

    model_name = model.__class__.__name__
    agent_name = model_name + "_" + env_name
    agent_folder_name = agent_name + "_" + start_time
    save_path = str(os.path.join(AGENTS_FOLDER, agent_folder_name))

    callback = CustomCheckpointCallback(
        save_freq=SAVING_FREQ,
        save_path=save_path,
        model_name=agent_name
    )

    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callback,
        log_interval=4,
        tb_log_name=agent_folder_name,
        progress_bar=True,
        reset_num_timesteps=False,
    )

    env.close()


def continue_training(algorithm):
    env = make_envs(ENV, NUM_ENVS)

    model = algorithm.load(
        path=CONTINUE_MODEL_PATH,
        env=env,
    )

    head, agent_checkpoint = os.path.split(CONTINUE_MODEL_PATH)
    agents_folder, agent_folder = os.path.split(head)
    agent_checkpoint_arr = agent_checkpoint.split("_")
    agent_name = "_".join(agent_checkpoint_arr[:-1])

    callback = CustomCheckpointCallback(
        save_freq=SAVING_FREQ,
        save_path=os.path.join(AGENTS_FOLDER, agent_folder),
        model_name=agent_name
    )

    model.learn(
        total_timesteps=TIMESTEPS,
        callback=callback,
        log_interval=4,
        tb_log_name=agent_folder,
        progress_bar=True,
        reset_num_timesteps=False
    )

    env.close()


if __name__ == '__main__':
    # train(ALGORITHM)
    # continue_training(ALGORITHM)

    for _env in [SC2LocalObservation5Env, SC2LocalObservation11Env, SC2LocalObservation17Env, SC2LocalObservationFlattenedEnv]:
        ENV = _env
        for i in range(5):
            train(ALGORITHM)
