import ImpalaCNN
from SuperMarioKartEnv import SuperMarioKartEnv
from NatureCNN import NatureCNN
from ImpalaCNN import ImpalaCNN

import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env import VecNormalize

from WandbTrajectoryCallback import WandbTrajectoryCallback


env = SuperMarioKartEnv(frameskipN=5)
model = PPO.load(
    "models/d8hmfozn/model.zip",
    env=env,
    device="cuda",
)

iteration = 0
for iteration in range(10):
    model.rollout_buffer.reset()



    while self.num_timesteps < total_timesteps:
        continue_training = self.collect_rollouts(self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps)

        if not continue_training:
            break

        iteration += 1
        self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

        # Display training infos
        if log_interval is not None and iteration % log_interval == 0:
            assert self.ep_info_buffer is not None
            self.dump_logs(iteration)

        self.train()

    callback.on_training_end()

    return self
