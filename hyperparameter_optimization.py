import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import psutil
import optuna
from main import make_env
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_param_importances, plot_slice
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from sc2_gym_wrapper import *

ENV = SC2GymEnvironment
ALGORITHM = DQN
N_TRIALS = 50


def optimize_dqn(trial):
    buffer_size = trial.suggest_categorical('buffer_size', [50_000, 100_000, 250_000, 500_000, 1_000_000])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2)
    gamma = trial.suggest_float('gamma', 0.90, 0.999)
    exploration_fraction = trial.suggest_float('exploration_fraction', 0.1, 0.5)

    num_envs = 6

    try:
        env = SubprocVecEnv([lambda: make_env(i) for i in range(num_envs)])
        env_names = env.get_attr("name")
        env_name = env_names[0]

        model = ALGORITHM(
            policy="MlpPolicy",
            env=env,
            buffer_size=buffer_size,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamma=gamma,
            exploration_fraction=exploration_fraction,
            verbose=0,
            device="cuda"
        )

        # Train and evaluate
        model.learn(total_timesteps=250_000)
        mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=20)

    except Exception as error:
        parent = psutil.Process(os.getpid())
        for child in parent.children(recursive=True):
            child.terminate()  # Gracefully terminate
        gone, alive = psutil.wait_procs(parent.children(), timeout=3)  # Wait for cleanup

        # If any processes are still alive, force kill them
        for child in alive:
            child.kill()

        return float("nan")

    env.close()

    return mean_reward


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.INFO)
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_dqn, n_trials=N_TRIALS)

    oputna_folder = "optuna"
    with open("hyperparameters.txt", "w") as file:
        print("Best hyperparameters:", study.best_params, file=file)
        optuna.visualization.plot_param_importances(study).write_html(os.path.join(oputna_folder, "param_importance.html"))
        optuna.visualization.plot_parallel_coordinate(study).write_html(os.path.join(oputna_folder, "parallel_coordinate.html"))
        optuna.visualization.plot_slice(study).write_html(os.path.join(oputna_folder, "slice_plot.html"))
        optuna.visualization.plot_contour(study).write_html(os.path.join(oputna_folder, "contour_plot.html"))
