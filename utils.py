import os
import psutil
import time

from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.monitor import Monitor


MONITOR_FOLDER = "monitor"
AGENTS_FOLDER = 'agents'


def make_monitored_env(env_class, start_time=None, env_id=0):
    env = env_class()
    filename = None
    if start_time is not None:
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


def make_envs(env_class, num_envs, start_time):
    env = None

    # Fixes weird Sc2 broken pipe error during init
    while env is None:
        try:
            # env = make_vec_env()
            env = SubprocVecEnv([lambda i=i: make_monitored_env(env_class, start_time, i) for i in range(num_envs)])

        except BrokenPipeError as error:
            env_error_cleanup()
            time.sleep(1)

    return env


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
