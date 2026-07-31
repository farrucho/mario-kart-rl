import ImpalaCNN
from SuperMarioKartEnv import SuperMarioKartEnv
from SuperMarioKartSelfPlayEnv import SuperMarioKartSelfPlayEnv
from NatureCNN import NatureCNN
from ImpalaCNN import ImpalaCNN

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback

from WandbTrajectoryCallback import WandbTrajectoryCallback

if __name__ == "__main__":
    config = {
        "total_timesteps": 100000000,
        "learning_rate": 2.5e-4,
        "n_steps": 2048,
        "batch_size": 2048,
        "n_epochs": 5,
        "gamma": 0.99,
        "clip_range": 0.2,
        "normalize_advantage": True,
        "ent_coef": 0.005,
        "vf_coef": 0.5,
        "max_grad_norm": 0.5,
        "target_kl": None,
        "gae_lambda": 0.95,
        "seed": 42,
        # stats
        "stats_window_size": 100,
        "model_save_freq": 30000,
        "gradient_save_freq": 40000,
        "record_video_trigger": 60000,
        "video_length": 1024, # igual ao nsteps para perceber
        "reward_log_freq": 5000, # global steps
        "trajectory_log_freq": 50000,
    }

    run = wandb.init(
        project="SuperMarioKart",
        name="Phase 2: (part4) 10 tracks uniform pool, added 5 more tracks because generalization was not good",
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

    checkpoint_cb = CheckpointCallback(
        save_freq=20000,
        save_path=f"models/{run.id}/checkpoints/",
        name_prefix="model",
        save_replay_buffer=False,
        save_vecnormalize=False
    )

    num_cpu = 10
    frameskip = 4
    framestack = 4

    selfplay = False

    if selfplay:
        models_pool = [
            "/home/farrucho/Desktop/extra/mario-kart-rl/models/s8x8ckan/model_30M.zip",
            "/home/farrucho/Desktop/extra/mario-kart-rl/models/uhwchs57/model_60M.zip",
            "/home/farrucho/Desktop/extra/mario-kart-rl/models/uhwchs57/model_60M.zip",
            "/home/farrucho/Desktop/extra/mario-kart-rl/models/f6ex7p8m/checkpoints/model_64024000_steps.zip",
            "/home/farrucho/Desktop/extra/mario-kart-rl/models/f6ex7p8m/checkpoints/model_65060000_steps.zip", 
            "/home/farrucho/Desktop/extra/mario-kart-rl/models/f6ex7p8m/checkpoints/model_66012000_steps.zip", 
            "/home/farrucho/Desktop/extra/mario-kart-rl/models/f6ex7p8m/checkpoints/model_66712000_steps.zip",
            "/home/farrucho/Desktop/extra/mario-kart-rl/models/cu1k1ouv/checkpoints/model_68070000_steps",
            "/home/farrucho/Desktop/extra/mario-kart-rl/models/cu1k1ouv/checkpoints/model_72060000_steps.zip",
            "/home/farrucho/Desktop/extra/mario-kart-rl/models/cu1k1ouv/checkpoints/model_70170000_steps.zip",
            "/home/farrucho/Desktop/extra/mario-kart-rl/models/cu1k1ouv/checkpoints/model_73740000_steps.zip",
        ]
        vec_env = SubprocVecEnv([lambda: Monitor(SuperMarioKartSelfPlayEnv(models_pool=models_pool, frameskipN=frameskip, frame_stack=framestack)) for _ in range(num_cpu)], start_method="spawn")
    else:
        vec_env = SubprocVecEnv([lambda cpu_n = cpu_n: Monitor(SuperMarioKartEnv(frameskipN=frameskip, trackstateindex=cpu_n, trainOnMultipleTracks=True)) for cpu_n in range(num_cpu)], start_method="fork")

    vec_env = VecFrameStack(vec_env, n_stack=framestack, channels_order="first")

    vec_env = VecVideoRecorder(
        vec_env,
        f"videos/{run.id}",
        record_video_trigger=lambda x: x % config["record_video_trigger"] == 0, # takes the current number of step
        video_length=config["video_length"], # Length of recorded videos
    )



    # model = PPO(
    #     "CnnPolicy",
    #     vec_env,
    #     policy_kwargs=
    #         dict(
    #             features_extractor_class=ImpalaCNN, # CHANGE THIS
    #         ),
    #     tensorboard_log=f"runs/{run.id}",
    #     verbose=1,
    #     device="cuda",
    #     learning_rate=config["learning_rate"],
    #     n_steps=config["n_steps"], # steps to run for each single environment in order to update
    #     # the model learns after the rollout buffer = n_steps * n_envs 
    #     batch_size=config["batch_size"], # minibatch size, the model looks at the rollout buffer, shuffles and divides into rollout_buffer/batch_size Batches, then updates for each batch
    #     n_epochs = config["n_epochs"],# perform the Batches update n_epoches times
    #     gamma = config["gamma"], # discount factor
    #     gae_lambda = config["gae_lambda"], # Smoothing factor for advantage estimation. Reduces variance in reward calculations.
    #     clip_range = config["clip_range"], # PPO core mechanic, the epsilon to limit how much the policy changes
    #     normalize_advantage = config["normalize_advantage"], # reduces variance
    #     ent_coef = config["ent_coef"], # Entropy coefficient for the loss calculation
    #     vf_coef = config["vf_coef"], # Value function coefficient for the loss calculation
    #     max_grad_norm = config["max_grad_norm"], # prevent exploding gradient, it clips them
    #     target_kl = config["target_kl"], # Limit the KL divergence between updates, because the clipping is not enough to prevent large. 
    #     stats_window_size = config["stats_window_size"], # how many episodes to get stats, for example mean reward,
    #     seed=config["seed"],
    # )


    # model.learn(
    #     total_timesteps=config["total_timesteps"],
    #     callback=[wandb_main_cb, traj_cb, checkpoint_cb],
    # )


    # LOAD FROM CHECKPOINT!!
    model = PPO.load(
        "models/6axs4z55/checkpoints/model_204400000_steps.zip",
        env=vec_env,
        device="cuda",
        total_timesteps = config["total_timesteps"],
        learning_rate = config["learning_rate"],
        n_steps = config["n_steps"],
        batch_size = config["batch_size"],
        n_epochs = config["n_epochs"],
        gamma = config["gamma"],
        clip_range = config["clip_range"],
        normalize_advantage = config["normalize_advantage"],
        ent_coef = config["ent_coef"],
        vf_coef = config["vf_coef"],
        max_grad_norm = config["max_grad_norm"],
        target_kl = config["target_kl"],
        gae_lambda = config["gae_lambda"],
        seed=config["seed"],
    )

    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[wandb_main_cb, traj_cb, checkpoint_cb],
        # callback=[wandb_main_cb, checkpoint_cb], 
        reset_num_timesteps=False,
    )



    # print(model.policy)
    run.finish()
