import os
import time
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from torch.utils.tensorboard import SummaryWriter


class GPUTensorBoardCallback(BaseCallback):
    def __init__(self):
        super().__init__()

        self.episode_rewards = []  # Store rewards for each episode
        self.current_episode_reward = 0  # Track accumulated reward in current episode
        self.episode_count = 0  # Count episodes

        # Create a new log directory for each run
        base_log_dir = "tensorboard"
        run_name = "agent_" + time.strftime("%d-%m-%Y_%H-%M-%S")
        log_dir = os.path.join(base_log_dir, run_name)

        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=log_dir)

    def _on_step(self) -> bool:

        # Track accumulated reward for the current episode
        if "rewards" in self.locals:
            self.current_episode_reward += sum(self.locals["rewards"]) / self.locals["self"].n_envs

        # Check if the episode is done
        if "dones" in self.locals and all(self.locals["dones"]):
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_count += 1

            # Compute the average accumulated reward over all episodes so far
            avg_accumulated_reward = sum(self.episode_rewards) / len(self.episode_rewards)
            self.writer.add_scalar("rollout/ep_rew_mean", avg_accumulated_reward, self.episode_count)

            # Reset episode reward for the next episode
            self.current_episode_reward = 0

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
