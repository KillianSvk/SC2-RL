import os
import time
import pysc2.bin.map_list

# from stable_baselines3 import DQN, PPO
# from sc2_environments import *
from utils import make_vec_env_sequential, get_latest_model_name, get_latest_model_checkpoint

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# NUM_ENVS = 6
# ENV = SC2LocalRoomsEnv
# ALGORITHM = DQN
# POLICY = "MlpPolicy" #MlpPolicy/CnnPolicy
# POLICY_KWARGS = dict(
#     # features_extractor_class=CustomizableCNN,
#     # features_extractor_kwargs=dict(features_dim=256),
#     # normalize_images=False,
#
#     # net_arch=[256, 256, 128]
#     # activation_fn=nn.ReLU
# )
# TIMESTEPS = 50_000
# SAVING_FREQ = 10_000

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
# tensorboard --logdir=tensorboard_defeat_zerg_bane
if __name__ == "__main__":
    # app.run(main)

    # env = ENV()
    # model = ALGORITHM(
    #     env=env,
    #     policy=POLICY,
    #     policy_kwargs=POLICY_KWARGS,
    #     tensorboard_log="tensorboard_defeat_zerg_bane",
    #     device="cuda",
    # )
    # print(model.policy)

    # from tbparse import SummaryReader
    #
    # reader = SummaryReader("tensorboard_multiprocess")
    # df = reader.scalars  # This gives you a DataFrame
    #
    # # Save to CSV
    # df.to_csv("tensorboard_multiprocess_data_indexed.csv")

    # env = ENV()
    # model = DQN(POLICY, env)
    # print(model.policy)

    latest_model_name = get_latest_model_name()
    latest_model_checkpoint = get_latest_model_checkpoint(latest_model_name)
    print(latest_model_name, latest_model_checkpoint)
