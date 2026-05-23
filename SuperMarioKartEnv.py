import stable_retro as retro
import stable_retro.data as stable_retro_data
import os
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class SuperMarioKartEnv(retro.RetroEnv):
    def __init__(self, frameskipN, render_mode="rgb_array"):
        super().__init__(game="/home/farrucho/Desktop/extra/mario-kart-rl/custom_integrations/SuperMarioKart-Snes", inttype=stable_retro_data.Integrations.ALL, render_mode=render_mode)
        
        self.observation, self.reward, self.terminated, self.info, = None,None,None,None
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=(1,84,84), # shape used in final pos-processing frame
            dtype=np.float32,
        )

        snes_buttons = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
        # Aquando fizer step o meu actions terá que seguir o seguinte template:
        self.allowed_buttons = ["B", "Y", "UP", "DOWN", "LEFT", "RIGHT", "A", "L", "R"]
        self.mapped_indices = np.array([snes_buttons.index(b) for b in self.allowed_buttons], dtype=np.uint8)
        
        self.action_space = gym.spaces.MultiBinary(len(self.mapped_indices))
        self.frameskipN = frameskipN

    def preprocessObs(self, observation):
        img = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[3:107,0:257] # top part of screen (104, 256)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        
        return img[np.newaxis, :, :].astype(np.float32) / 255.0

    def step(self, action):
        full_actions = np.zeros(12, dtype=np.uint8)
        full_actions[self.mapped_indices] = action

        reward = 0
        for j in range(0, self.frameskipN):
            next_observation, next_reward, next_terminated, _, next_info = super().step(full_actions)
            if next_terminated:
                break
        
        next_observation = self.preprocessObs(next_observation)

        if self.info == None or self.info == {}:
            # print("TRIGGER, FIRST STEP")
            next_info['position_history_south'] = np.array((), dtype=np.int16)
            next_info['position_history_east'] = np.array((), dtype=np.int16)
            next_info['last_checkpoint_time'] = 0

            next_info['checkpoints crossed'] = {
                _: set() for _ in range(0,7) 
            }

            next_info['reward_checkpoint'] = 0
            next_info['reward_lap'] = 0
            next_info['reward_collision'] = 0
            next_info['reward_rank'] = 0
            next_info['reward_speed'] = 0
            next_info['reward_finish_race'] = 0
            next_info['reward_time'] = 0



            self.observation, self.reward, self.terminated, self.info, = next_observation, next_reward, False, next_info        


            return next_observation, reward, next_terminated, False, next_info
        else:
            # store info
            next_info['position_history_south'] = np.append(self.info['position_history_south'], next_info['pos_south'])
            next_info['position_history_east'] = np.append(self.info['position_history_east'], next_info['pos_east'])
            next_info['last_checkpoint_time'] = self.info['last_checkpoint_time']
            next_info['checkpoints crossed'] = self.info['checkpoints crossed']

            next_info['reward_checkpoint'] = self.info['reward_checkpoint']
            next_info['reward_lap'] = self.info['reward_lap']
            next_info['reward_collision'] = self.info['reward_collision']
            next_info['reward_rank'] = self.info['reward_rank']
            next_info['reward_speed'] = self.info['reward_speed']
            next_info['reward_finish_race'] = self.info['reward_finish_race']
            next_info['reward_time'] = self.info['reward_time']


            next_lap = next_info['current_lap']-127 # 0,1,2,3,4,5,6==finished
            next_rank = next_info['rank']/2+1 # (Rank-1)*2)
            next_checkpoint = next_info['checkpoint']
            next_seconds = next_info['clock_minutes']*60 + next_info['clock_seconds']
            next_collision = next_info['collision_detection']


            current_lap = self.info['current_lap']-127 # 0,1,2,3,4,5,6==finished
            current_max_reached_lap = self.info['max_reached_lap']-127 
            current_rank = self.info['rank']/2+1 # (Rank-1)*2)
            current_checkpoint = self.info['checkpoint']
            current_seconds = self.info['clock_minutes']*60 + self.info['clock_seconds']

            # passou um checkpoint e prevenir que nao ande às voltas
            if next_checkpoint > current_checkpoint:
                if next_checkpoint not in next_info['checkpoints crossed'][next_lap]:
                    reward += 0.5
                    next_info['checkpoints crossed'][next_lap].add(next_checkpoint)
                    next_info['reward_checkpoint'] += 0.5
                    next_info['last_checkpoint_time'] = next_seconds

            if next_lap > current_lap and next_lap != current_max_reached_lap and next_lap != 1: # check se realmente passou para uma nova lap e nao da uma volta inicial assim que a corrida comeca
                # passou uma lap
                reward += 100
                next_info['reward_lap'] += 100

            
            if next_collision == 7:
                # penalizar colisoes
                reward += -5
                next_info['reward_collision'] += -5
                # print("colisao")

            # rank reward per step
            reward += (0.1*1/next_rank - 0.1*1/8)
            next_info['reward_rank'] += (0.1*1/next_rank - 0.1*1/8) 

            # promover velocidade, encorajar para aprender mais cedo a nao ficar parado
            # module de cada componente da velocidade vai de de 0 até 1000, 2000 no maximo se usar cogumelo
            reward += 0.0001*(np.sqrt(next_info['speed_south']**2 + next_info['speed_east']**2)-600) # reward negativa se andar devagar
            next_info['reward_speed'] += 0.0001*np.sqrt(next_info['speed_south']**2 + next_info['speed_east']**2)
            # reward finishing race
            if next_info['current_lap'] == 133:
                reward += 100
                next_info['reward_finish_race'] += 100


            # com frameskip de 4 obtém 1569 steps -> ~10steps/segundo
            terminated = (next_info['current_lap'] == 133) or next_seconds - next_info['last_checkpoint_time'] > 15 or (next_seconds > 150) 
            # terminar se user acabou corrida ou nao houver progressao checkpoint ou episodio > 150s
            
            # time penalty at the end of episode
            if terminated:
                # print(next_seconds) # at max it will be 94
                reward += -0.5*next_seconds
                next_info['reward_time'] += -0.5*next_seconds


        self.observation, self.reward, self.terminated, self.info = next_observation, reward, terminated, next_info
        return self.observation, self.reward, self.terminated, _, self.info

    def reset(self, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        preprocessed_obs = self.preprocessObs(observation)
        

        self.observation = preprocessed_obs
        self.info = info
        self.reward = 0
        self.terminated = False
        return preprocessed_obs, info


    def save_png_observation(self, observation, filepath="./images/test_observation.png"):
        img = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = img[3:107,0:257] # top part of screen (104, 256)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        cv2.imwrite(filepath, img)


# env = SuperMarioKartEnv(frameskipN=10, render_mode="human")
# env.reset()
# terminated = False
# while not terminated:
#     action = env.action_space.sample()
#     action[0] = 1
#     obs, reward, terminated , _, info = env.step(action)


# import matplotlib.pyplot as plt
# from stable_baselines3.common.monitor import Monitor
# from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack


# if __name__ == '__main__':
#     # Initialize the VecEnv as before...
#     vec_env = SubprocVecEnv([lambda: Monitor(SuperMarioKartEnv(frameskipN=5)) for _ in range(4)], start_method="spawn")
#     vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")

#     # 1. Reset
#     obs = vec_env.reset()

#     print("Performing 500 random steps...")
#     for _ in range(100):
#         # Generate random actions for all 12 envs
#         # Note: vec_env.action_space.sample() returns a batch if it's a VecEnv
#         actions = [[1,0,0,0,0,0,0,0,0] for _ in range(vec_env.num_envs)]
        
#         actions[0][0] = 1 if np.random.rand() > 0.01 else 0

#         # Unpack into 4 variables: obs, rewards, dones, infos
#         obs, rewards, dones, infos = vec_env.step(actions)
#     print("Plotting observations for all 12 environments...")

#     # obs shape is (12, 4, 84, 84)
#     # Create a grid: 12 rows (environments) x 4 columns (frames)
#     fig, axes = plt.subplots(4, 4, figsize=(12, 24))

#     for env_idx in range(4):
#         for frame_idx in range(4):
#             # obs[env_idx] is the stack (4, 84, 84)
#             # frame is (84, 84)
#             frame = obs[env_idx, frame_idx, :, :]
            
#             axes[env_idx, frame_idx].imshow(frame, cmap='gray', vmin=0, vmax=1)
#             axes[env_idx, frame_idx].axis('off')
            
#             # Label the columns
#             if env_idx == 0:
#                 axes[env_idx, frame_idx].set_title(f"Frame {frame_idx}")
        
#         # Label the rows
#         axes[env_idx, 0].text(-10, 42, f"Env {env_idx}", va='center', ha='right', fontweight='bold')

#     plt.tight_layout()
#     plt.show()
