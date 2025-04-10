import optuna
from stable_baselines3 import DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

from sc2_gym_wrapper import *

ENV = SC2GymEnvironment
ALGORITHM = DQN


def make_env():
    return ENV()


def optimize_dqn(trial):
    buffer_size = trial.suggest_categorical('buffer_size', [50_000, 100_000, 250_000, 500_000, 1_000_000])
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    gamma = trial.suggest_uniform('gamma', 0.90, 0.999)
    exploration_fraction = trial.suggest_uniform('exploration_fraction', 0.1, 0.5)

    num_envs = 6
    env = SubprocVecEnv([lambda: make_env() for _ in range(num_envs)])
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
    model.learn(total_timesteps=50_000)
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
    env.close()

    return mean_reward


if __name__ == '__main__':
    study = optuna.create_study(direction='maximize')
    study.optimize(optimize_dqn, n_trials=50)

    print("Best hyperparameters:", study.best_params)
