import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class ResidualBlock(nn.Module):
    def __init__(self, channels) -> None:
        super().__init__()

        self.block= nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, padding=1),
        )

    def forward(self,x):
        return x + self.block(x)


class ImpalaBlock(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=3,
            stride=1,
            padding=1
        )

        self.maxPool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.res1 = ResidualBlock(out_channels)
        self.res2 = ResidualBlock(out_channels)
    
    def forward(self,x):
        x = self.conv(x)
        x = self.maxPool(x)
        x = self.res1(x)
        x = self.res2(x)
        return x



class ImpalaCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # features dim is the final fully connected layer logits
        # We assume CxHxW images (channels first)
        n_input_channels = observation_space.shape[0]

        self.cnn = nn.Sequential(
            ImpalaBlock(n_input_channels, 16), # 1x84x84 ->conv -> 16x84x84 -> maxpool -> 16x42x42 -> residual -> 16x42x42 -> residual -> 16x42x42
            ImpalaBlock(16,32),
            ImpalaBlock(32,32),
            nn.ReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim), 
                nn.ReLU()
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations.float()/255.0))