import os
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

    if algorithm is None:
        print("Wrong or None algorithm was chosen!")
        return

    if argv[2] == 'train':
        train(algorithm)

    elif argv[2] == 'test':
        test(algorithm)


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
