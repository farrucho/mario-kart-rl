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

from CustomCNN import CustomCNN


from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecNormalize

from WandbTrajectoryCallback import WandbTrajectoryCallback

if __name__ == "__main__":
    config = {
        "policy_type": "CnnPolicy",
        "total_timesteps": 100000000,
        "learning_rate": 2.5e-4,
        "n_steps": 1024,
        "batch_size": 256,
        "n_epochs": 5,
        "gamma": 0.99,
        "clip_range": 0.2,
        "normalize_advantage": True,
        "ent_coef": 0.02,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        # stats
        "stats_window_size": 100,
        "model_save_freq": 40000,
        "gradient_save_freq": 40000,
        "record_video_trigger": 8000,
        "video_length": 1024, # igual ao nsteps para perceber
        "reward_log_freq": 1000, # global steps
        "trajectory_log_freq": 10000,
    }

    run = wandb.init(
        project="SuperMarioKart",
        name="changed speed reward, trying to get good rankings",
        config=config,
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=True,  # auto-upload the videos of agents playing the game
        save_code=True,  # optional
    )

    wandb_main_cb = WandbCallback(
            model_save_freq=config["model_save_freq"],
            gradient_save_freq=config["gradient_save_freq"],
            model_save_path=f"models/{run.id}",
            verbose=2,
        )

    traj_cb = WandbTrajectoryCallback(reward_log_freq=config["reward_log_freq"],
    trajectory_log_freq=config["trajectory_log_freq"])


    num_cpu = 12

    vec_env = SubprocVecEnv([lambda: Monitor(SuperMarioKartEnv(frameskipN=5)) for _ in range(num_cpu)], start_method="spawn")
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")

    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=True,
        clip_reward=10.0,
    )

    # VIDEO RECORDER
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
    #     callback=[wandb_main_cb, traj_cb],
    # )


    # LOAD FROM CHECKPOINT!!
    model = PPO.load(
        "models/4lpio9u4/model.zip",
        env=vec_env,
        device="cuda",
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[wandb_main_cb, traj_cb],
        reset_num_timesteps=False,
    )



    run.finish()
