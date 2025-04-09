import os
import time
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class MultiprocessTensorBoardCallback(BaseCallback):
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

        self.start_timesteps = self.num_timesteps
    def on_training_start(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        super().on_training_start(locals_, globals_)

    def on_rollout_start(self) -> None:
        super().on_rollout_start()

    def log_fps(self):
        elapsed_time = time.time() - self.locals["self"].start_time
        if elapsed_time > 0:
            # Compute FPS (environment steps per second)
            total_steps = (self.num_timesteps - self.start_timesteps) * self.locals["self"].n_envs
            fps = total_steps / elapsed_time

            # Compute Iterations per second
            iterations_per_sec = self.locals["num_collected_steps"] / elapsed_time

            # Log values
            self.logger.record("train/FPS", fps)
            self.logger.record("train/iterations_per_sec", iterations_per_sec)

    def log_episode_reward_mean(self):
        rewards = self.locals["rewards"]
        dones = self.locals["dones"]

        current_episode_reward = sum(self.locals["rewards"]) / self.locals["self"].n_envs
        self.logger.record_mean("rollout/episode_reward_mean", current_episode_reward)

    def on_step(self) -> bool:
        super()._on_step()

        # self.num_timesteps
        #
        # self.locals["self"].exploration_rate
        # self.locals["self"].learning_rate
        # self.locals["self"].num_timesteps
        #
        # self.locals["rewards"]
        # self.locals["dones"]
        # self.locals['tb_log_name'] #DQN
        # self.locals['reset_num_timesteps']
        # self.locals["num_collected_steps"]
        # self.locals["num_collected_episodes"]

        self.log_fps()

        if "rewards" in self.locals:
            self.log_episode_reward_mean()

        if hasattr(self.model, "exploration_rate"):
            self.logger.record_mean("rollout/exploration_rate", self.model.exploration_rate)

        # Log training loss if present
        if "loss" in self.locals:
            loss = self.locals["loss"]
            if hasattr(loss, "item"):
                loss = loss.item()
            self.logger.record_mean("train/loss", loss)

        # Optionally log learning rate
        if hasattr(self.model, "learning_rate"):
            lr = self.model.learning_rate
            if callable(lr):
                lr = lr(self.num_timesteps)
            self.logger.record_mean("train/learning_rate", lr)

        self.logger.log()
        # self.logger.dump(self.num_timesteps)

        return True

    def on_rollout_end(self) -> None:
        super().on_rollout_end()

    def on_training_end(self) -> None:
        super().on_training_end()


class CustomMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

        self.episode_rewards = []
        self.current_rewards = None
        self.start_time = None
        self.total_elapsed_time = 0.0  # Total across multiple learn() calls

    def _on_training_start(self) -> None:
        self.current_rewards = [0.0 for _ in range(self.model.get_env().num_envs)]
        self.start_time = time.time()

    def record_episode_reward_mean(self):
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        if rewards is not None and dones is not None:
            for i, (r, done) in enumerate(zip(rewards, dones)):
                self.current_rewards[i] += r
                if done:
                    self.episode_rewards.append(self.current_rewards[i])
                    self.current_rewards[i] = 0

            if self.episode_rewards:
                avg_rew = sum(self.episode_rewards[-100:]) / min(len(self.episode_rewards), 100)
                self.logger.record("rollout/episode_reward_mean", avg_rew)

    def _on_step(self) -> bool:

        self.record_episode_reward_mean()

        elapsed = time.time() - self.start_time
        self.logger.record("time/elapsed_training_time_sec", elapsed)

        return True

    def _on_training_end(self) -> None:
        elapsed = time.time() - self.start_time
        self.total_elapsed_time += elapsed
        self.logger.record("time/total_training_time_sec", self.total_elapsed_time)
        
