import os
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import optuna
import torch.nn as nn
from optuna.visualization import plot_optimization_history, plot_parallel_coordinate, plot_param_importances, plot_slice, plot_contour
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env

from sc2_environments import *
from utils import make_monitored_env, make_envs
from custom_features import CustomizableCNN

OPTUNA_FOLDER = "optuna_screen"
NUM_ENVS = 6
ENV = SC2ScreenEnv
ALGORITHM = DQN
POLICY = "CnnPolicy"
TIMESTEPS_PER_MODEL = 250_000
N_TRIALS = 15
N_STARTUP_TRIALS = 5
N_EVAL_EPISODES = 100

# ENV = SC2ScreenEnv, TIMESTEPS_PER_MODEL = 250_000, N_TRIALS = 15 => ~7h

def optimize_dqn(trial):

    gamma = trial.suggest_float('gamma', 0.90, 0.999)
    max_gradient_norm = trial.suggest_float("max_gradient_norm", 0.3, 20.0, log=True)
    buffer_size = trial.suggest_categorical('buffer_size', [50_000, 100_000, 250_000, 500_000, 1_000_000])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1, log=True)
    exploration_fraction = trial.suggest_float('exploration_fraction', 0.1, 0.5)
    target_update_interval = trial.suggest_categorical("target_update_interval", [(x+1) * 1000 for x in range(10)])
    gradient_steps = trial.suggest_int("gradient_steps", 1, 64)

    # CNN architecture hyperparameters
    # cnn_kwargs = dict()
    # net_arch_tiny = {
    #     "pi": [64],
    #     "vf": [64],
    # }
    # net_arch_default = {
    #     "pi": [64, 64],
    #     "vf": [64, 64],
    # }
    # net_arch_big = {
    #     "pi": [64, 64, 64],
    #     "vf": [64, 64, 64],
    # }
    #
    # cnn_kwargs["net_arch"] = trial.suggest_categorical("net_arch", [net_arch_tiny, net_arch_default, net_arch_big])
    # cnn_kwargs["activation_fn"] = trial.suggest_categorical("activation_fn", [nn.Tanh, nn.ReLU,])

    env = make_envs(ENV, NUM_ENVS)

    model = ALGORITHM(
        env=env,
        policy=POLICY,
        # policy_kwargs=cnn_kwargs,

        buffer_size=buffer_size,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        gradient_steps=gradient_steps,
        max_grad_norm=max_gradient_norm,
        exploration_fraction=exploration_fraction,
        target_update_interval=target_update_interval,

        verbose=0,
        device="cuda"
    )

    model.learn(total_timesteps=TIMESTEPS_PER_MODEL)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=N_EVAL_EPISODES)

    env.close()
    return mean_reward


if __name__ == '__main__':
    optuna.logging.set_verbosity(optuna.logging.INFO)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)
    # Do not prune before 1/3 of the max budget is used
    pruner = MedianPruner(n_startup_trials=N_STARTUP_TRIALS, n_warmup_steps=TIMESTEPS_PER_MODEL // 3)

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    study.optimize(optimize_dqn, n_trials=N_TRIALS)

    os.makedirs(OPTUNA_FOLDER, exist_ok=True)

    with open(os.path.join(OPTUNA_FOLDER, "hyperparameters.txt"), "w") as file:
        # print(f"Best trial values: {study.best_value} \n hyperparameters: {study.best_params}", file=file)

        print("Best trial:", file=file)
        best_trial = study.best_trial

        print(f"  Value: {best_trial.value}", file=file)

        print("  Params: ")
        for key, value in best_trial.params.items():
            print(f"    {key}: {value}", file=file)

    study.trials_dataframe().to_csv(os.path.join(OPTUNA_FOLDER, "study_results.csv"))

    plot_optimization_history(study).write_html(os.path.join(OPTUNA_FOLDER, "plot_optimization_history.html"))
    plot_param_importances(study).write_html(os.path.join(OPTUNA_FOLDER, "param_importance.html"))
    plot_parallel_coordinate(study).write_html(os.path.join(OPTUNA_FOLDER, "parallel_coordinate.html"))
    plot_slice(study).write_html(os.path.join(OPTUNA_FOLDER, "slice_plot.html"))
    plot_contour(study).write_html(os.path.join(OPTUNA_FOLDER, "contour_plot.html"))
