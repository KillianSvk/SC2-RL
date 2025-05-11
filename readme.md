# Setup

Download Starcraft II: \
&nbsp;&nbsp;&nbsp;&nbsp; Windows: https://starcraft2.blizzard.com/en-us/ \
&nbsp;&nbsp;&nbsp;&nbsp; Linux: https://github.com/Blizzard/s2client-proto#downloads (recommended version is 4.10) 

Set enviroment variable SC2PATH on your OS to your Starcraft II directory

Recommended python version 3.10 \
Recommended to use virtual environment for managing dependencies 

For GPU acceleration: \
&nbsp;&nbsp;&nbsp;&nbsp; Recommended cuda version 12.6 \
&nbsp;&nbsp;&nbsp;&nbsp; Download pytorch (pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126) 

donwloading dependencies from [requirements](requirements.txt)

# [Training](train.py)
ENV is environment used for learning \
NUM_ENVS number of vectorized environments \
ALGORITHM algorithms from SB3 used for learning \
POLICY algorithm policy (MlpPolicy/CnnPolicy/MultiInputPolicy) \
POLICY_KWARGS policy arguments \
TIMESTEPS how many timesteps used for learning \
SAVING_FREQ how many timesteps between agent saving checkpoints

run train.py

if you want to continue training from checkpoint, change main to use continue_training function and 
set model_path in that function to desired path to checkpoint

# [Testing](test.py)
ENV is environment used for testing \
ALGORITHM algorithms from SB3 used for learning the model\
NUM_TESTING_EPISODES number of episodes used for testing \
MODEL_PATH is a path to the model checkpoint (can use get_latest_model_path for latest model)

run test.py


