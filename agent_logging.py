import os
import time
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from torch.utils.tensorboard import SummaryWriter


class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq, save_path, model_name, verbose=0):
        super().__init__(verbose)

        self.save_freq = save_freq
        self.save_path = save_path
        self.model_name = model_name

        self.start_time = None
        self.time_elapsed = 0
        self.episode_rewards = None

    def _init_callback(self) -> None:
        os.makedirs(self.save_path, exist_ok=True)

    def on_training_start(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        super().on_training_start(locals_, globals_)

        self.episode_rewards = [0.0 for _ in range(self.model.get_env().num_envs)]

        self.start_time = time.perf_counter()
        if hasattr(self.logger, "name_to_value") and "time/training_time" in self.logger.name_to_value:
            self.time_elapsed = self.logger.name_to_value["time/training_time"]

    def _on_rollout_start(self) -> None:
        super()._on_rollout_start()

    def record_episode_reward_mean(self):
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")

        for i, (reward, done) in enumerate(zip(rewards, dones)):
            self.episode_rewards[i] += reward
            self.logger.record_mean("rollout/episode_reward_mean", self.episode_rewards[i])

            if done:
                # average_episode_reward = sum(self.episode_rewards) / self.locals["self"].n_envs
                # self.logger.record_mean("rollout/episode_reward_mean", average_episode_reward)
                # for i in range(len(self.episode_rewards)):
                self.episode_rewards[i] = 0

    def record_time_elapsed(self):
        elapsed = time.perf_counter() - self.start_time
        self.logger.record("time/training_time", self.time_elapsed + elapsed)

    def save_model(self):
        filename = os.path.join(self.save_path, f"{self.model_name}_{self.num_timesteps // 1_000}k.zip")
        self.model.save(path=filename)

    def _on_step(self) -> bool:
        super()._on_step()

        # self.record_episode_reward_mean()

        # self.record_time_elapsed()

        if self.locals["self"].n_envs > (self.num_timesteps % self.save_freq) >= 0 and self.num_timesteps != 0:
            self.save_model()

        # now_time = time.time()
        # elapsed = now_time - self.start_time
        # self.logger.record("time/time_elapsed", elapsed)

        return True

    def _on_rollout_end(self) -> None:
        super()._on_rollout_end()

    def _on_training_end(self) -> None:
        super()._on_training_end()

        self.save_model()
