import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class CustomSmallerCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box):
        super().__init__(observation_space, features_dim=128)  # final output size
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )
        # Compute the flattened output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 36, 36)
            n_flatten = self.cnn(dummy_input).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, 128), nn.ReLU())

    def forward(self, x):
        return self.linear(self.cnn(x))

class CustomizableCNN(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 512,
        kernel_sizes: tuple = (8, 4, 3),
        strides: tuple = (4, 2, 1),
        filters: tuple = (32, 64, 64),
        use_batch_norm: bool = False
    ):
        super().__init__(observation_space, features_dim)

        assert len(kernel_sizes) == 3 and len(strides) == 3 and len(filters) == 3, \
            "kernel_sizes, strides, and filters must each be of length 3"

        n_input_channels = observation_space.shape[0]
        layers = []

        in_channels = n_input_channels
        for i in range(3):
            layers.append(
                nn.Conv2d(in_channels, filters[i], kernel_size=kernel_sizes[i], stride=strides[i])
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(filters[i]))
            layers.append(nn.ReLU())
            in_channels = filters[i]

        layers.append(nn.Flatten())
        self.cnn = nn.Sequential(*layers)

        # Dynamically compute output size
        with torch.no_grad():
            sample_input = torch.zeros(1, *observation_space.shape)
            cnn_output_dim = self.cnn(sample_input).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(cnn_output_dim, features_dim),
            nn.ReLU()
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(obs))
