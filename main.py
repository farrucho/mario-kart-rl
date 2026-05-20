import stable_retro as retro
import stable_retro.data as stable_retro_data
import os
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import pdb

from SuperMarioKartEnv import SuperMarioKartEnv


from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack


import torch as th
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomCNN(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        # We assume CxHxW images (channels first)
        # Re-ordering will be done by pre-preprocessing or wrapper
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        # pdb.set_trace()
        # Compute shape by doing one forward pass
        with th.no_grad():
            n_flatten = self.cnn(
                th.as_tensor(observation_space.sample()[None]).float()
            ).shape[1]

        self.linear = nn.Sequential(
                nn.Linear(n_flatten, features_dim), 
                nn.ReLU()
            )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))




from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor

if __name__ == "__main__":
    config = {
        "policy_type": "CnnPolicy",
        "total_timesteps": 10000000,
        "learning_rate":0.0003,
        "n_steps":1024,
        "batch_size": 64,
        "n_epochs": 10,
        "gamma": 0.995,
        "clip_range": 0.2,
        "normalize_advantage": True,
        "ent_coef": 0.01,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        # stats
        "stats_window_size": 100,
        "model_save_freq": 40000,
        "gradient_save_freq": 40000,
        "record_video_trigger": 40000,
        "video_length": 1024 # igual ao nsteps para perceber
    }

    run = wandb.init(
        project="SuperMarioKart",
        name="(part5) stats_window_size changed to 100 episodes - 8 workers and spawn method",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    num_cpu = 12

    vec_env = SubprocVecEnv([lambda: Monitor(SuperMarioKartEnv()) for _ in range(num_cpu)], start_method="spawn")
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")
    vec_env = VecVideoRecorder(
        vec_env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % config["record_video_trigger"] == 0, # takes the current number of step
        video_length=config["video_length"], # Length of recorded videos
    )

    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=128),
    )

    # model = PPO(
    #     "CnnPolicy",
    #     vec_env,
    #     policy_kwargs=policy_kwargs,
    #     tensorboard_log=f"runs/{run.id}",
    #     verbose=1,
    #     device="cuda",
    #     learning_rate=config["learning_rate"],
    #     n_steps=config["n_steps"], # steps to run for each single environment in order to update
    #     # the model learns after the rollout buffer = n_steps * n_envs 
    #     batch_size=config["batch_size"], # minibatch size, the model looks at the rollout buffer, shuffles and divides into rollout_buffer/batch_size Batches, then updates for each batch
    #     n_epochs = config["n_epochs"],# perform the Batches update n_epoches times
    #     gamma = config["gamma"], # discount factor
    #     # gae_lambda = 0.95 Smoothing factor for advantage estimation. Reduces variance in reward calculations.
    #     clip_range = config["clip_range"], # PPO core mechanic, the epsilon to limit how much the policy changes
    #     normalize_advantage = config["normalize_advantage"], # reduces variance
    #     ent_coef = config["ent_coef"], # Entropy coefficient for the loss calculation
    #     vf_coef = config["vf_coef"], # Value function coefficient for the loss calculation
    #     max_grad_norm = config["max_grad_norm"], # prevent exploding gradient, it clips them
    #     target_kl = config["target_kl"], # Limit the KL divergence between updates, because the clipping is not enough to prevent large. 
    #     stats_window_size = config["stats_window_size"], # how many episodes to get stats, for example mean reward
    # )

    # model.learn(
    #     total_timesteps=config["total_timesteps"],
    #     callback=WandbCallback(
    #         model_save_freq=config["model_save_freq"],
    #         gradient_save_freq=config["gradient_save_freq"],
    #         model_save_path=f"models/{run.id}",
    #         verbose=2,
    #     ),
    # )


    # LOAD FROM CHECKPOINT!!
    model = PPO.load(
        "models/fptsne5c/model.zip",
        env=vec_env,
        device="cuda",
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            model_save_freq=config["model_save_freq"],
            gradient_save_freq=config["gradient_save_freq"],
            model_save_path=f"models/{run.id}",
            verbose=2,
        ),
        reset_num_timesteps=False,
    )



    run.finish()
