import os
import time
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class MultiprocessTensorBoardCallback(BaseCallback):
    def __init__(self):
        super().__init__()

        self.episode_rewards = []  # Store rewards for each episode
        self.current_episode_reward = 0  # Track accumulated reward in current episode
        self.writer = None

    def on_training_start(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        super().on_training_start(locals_, globals_)

        base_log_dir = "tensorboard"
        if self.num_timesteps > 0:
            existing_runs = sorted(
                os.listdir(base_log_dir),
                key=lambda x: os.path.getctime(os.path.join(base_log_dir, x)),
                reverse=True
            )
            last_run = existing_runs[0]
            log_dir = os.path.join(base_log_dir, last_run)

        else:
            # Create a new log directory for each run
            env_names = self.model.env.get_attr("name")
            model_name = env_names[0]
            run_name = model_name + "_" + time.strftime("%H-%M-%S")
            log_dir = os.path.join(base_log_dir, run_name)

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def _on_step(self) -> bool:

        # Track accumulated reward for the current episode
        if "rewards" in self.locals:
            self.current_episode_reward += sum(self.locals["rewards"]) / self.locals["self"].n_envs

        # Check if the episode is done
        if "dones" in self.locals and any(self.locals["dones"]):
            self.episode_rewards.append(self.current_episode_reward)

            # Reset episode reward for the next episode
            self.current_episode_reward = 0

        if self.episode_rewards:
            episode_reward_mean = (self.current_episode_reward + sum(self.episode_rewards)) / (len(self.episode_rewards) + 1)
            self.writer.add_scalar("rollout/ep_reward_mean", episode_reward_mean, self.num_timesteps)

        if "rewards" in self.locals and len(self.locals["rewards"]) > 0:
            avg_step_reward = np.mean(self.locals["rewards"])
            self.writer.add_scalar("rollout/step_reward_mean", avg_step_reward, self.num_timesteps)

        # Log exploration rate if available
        if hasattr(self.model, "exploration_rate"):
            self.writer.add_scalar("rollout/exploration_rate", self.model.exploration_rate, self.num_timesteps)

        # Log key training metrics
        if "loss" in self.locals:
            loss = self.locals["loss"].item()  # Extract loss value
            self.writer.add_scalar("train/loss", loss, self.num_timesteps)

        # Log Q-values
        # if hasattr(self.model, "q_net"):
        #     q_values = self.model.q_net(self.model.policy.q_net_input)
        #     mean_q_value = torch.mean(q_values).item()
        #     self.writer.add_scalar("train/q_values", mean_q_value, self.num_timesteps)

        if hasattr(self.model, "learning_rate"):
            self.writer.add_scalar("train/learning_rate", self.model.learning_rate, self.num_timesteps)

        return True

    def _on_training_end(self) -> None:
        self.writer.close()
