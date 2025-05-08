import os

import pysc2.bin.map_list
from absl import flags, app

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from stable_baselines3 import DQN, PPO

from sc2_environments import *
from train import train
from test import test

FLAGS = flags.FLAGS

AGENTS_FOLDER = 'agents'
MONITOR_FOLDER = "monitor"

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


def run_from_cmd(argv):
    algorithm = None

    if argv[1] == 'dqn':
        algorithm = DQN

    elif argv[1] == 'ppo':
        algorithm = PPO

    else:
        print("Wrong or None algorithm was chosen!")
        return

    if argv[2] == 'train':
        train(algorithm)

    elif argv[2] == 'test':
        test(algorithm)

    else:
        print('Write "test" or "train" for training or testing')


def main(argv):
    if len(argv) == 2:
        run_from_cmd(argv)


# triton==3.2.0
# nvidia-cublas-cu12==12.4.5.8
# nvidia-cuda-cupti-cu12==12.4.127
# nvidia-cuda-nvrtc-cu12==12.4.127
# nvidia-cuda-runtime-cu12==12.4.127
# nvidia-cudnn-cu12==9.1.0.70
# nvidia-cufft-cu12==11.2.1.3
# nvidia-curand-cu12==10.3.5.147
# nvidia-cusolver-cu12==11.6.1.9
# nvidia-cusparse-cu12==12.3.1.170
# nvidia-cusparselt-cu12==0.6.2
# nvidia-nccl-cu12==2.21.6
# nvidia-nvjitlink-cu12==12.4.127
# nvidia-nvtx-cu12==12.4.127

# scp -r C:\Users\petoh\Desktop\School\Bakalarka\web\index.html hozlar5@davinci.fmph.uniba.sk:~/public_html/bakalarska_praca/
# tensorboard --logdir=tensorboard
if __name__ == "__main__":
    app.run(main)

    # env = ENV()
    # model = ALGORITHM(
    #     env=env,
    #     policy=POLICY,
    #     policy_kwargs=POLICY_KWARGS,
    #     tensorboard_log="tensorboard",
    #     device="cuda",
    # )
    # print(model.policy)
