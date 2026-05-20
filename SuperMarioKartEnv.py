import stable_retro as retro
import stable_retro.data as stable_retro_data
import os
import cv2
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class SuperMarioKartEnv(retro.RetroEnv):
    def __init__(self, render_mode="rgb_array"):
        super().__init__(game="/home/farrucho/Desktop/extra/mario-kart-rl/custom_integrations/SuperMarioKart-Snes", inttype=stable_retro_data.Integrations.ALL, render_mode=render_mode)
        
        self.observation, self.reward, self.terminated, self.info, = None,None,None,None
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(1,84,84), # shape used in final pos-processing frame
            dtype=np.float32,
        )

        snes_buttons = ["B", "Y", "SELECT", "START", "UP", "DOWN", "LEFT", "RIGHT", "A", "X", "L", "R"]
        # Aquando fizer step o meu actions terá que seguir o seguinte template:
        self.allowed_buttons = ["B", "Y", "UP", "DOWN", "LEFT", "RIGHT", "A", "L", "R"]
        self.mapped_indices = np.array([snes_buttons.index(b) for b in self.allowed_buttons], dtype=np.uint8)
        
        self.action_space = gym.spaces.MultiBinary(len(self.mapped_indices))

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
        next_observation, next_reward, next_terminated, _, next_info = super().step(full_actions)
        
        next_observation = self.preprocessObs(next_observation)

        if self.info == None or self.info == {}:
            # print("TRIGGER, FIRST STEP")
            next_info['position_history_south'] = np.array((), dtype=np.int16)
            next_info['position_history_east'] = np.array((), dtype=np.int16)

            self.observation, self.reward, self.terminated, self.info, = next_observation, next_reward, False, next_info        


            return next_observation, reward, next_terminated, False, next_info
        else:
            # store info
            next_info['position_history_south'] = np.append(self.info['position_history_south'], next_info['pos_south'])
            next_info['position_history_east'] = np.append(self.info['position_history_east'], next_info['pos_east'])

            next_lap = next_info['current_lap']-127 # 0,1,2,3,4,5,6==finished
            next_rank = next_info['rank']/2+1 # (Rank-1)*2)
            next_checkpoint = next_info['checkpoint']
            next_seconds = next_info['clock_minutes']*60 + next_info['clock_seconds']


            current_lap = self.info['current_lap']-127 # 0,1,2,3,4,5,6==finished
            current_rank = self.info['rank']/2+1 # (Rank-1)*2)
            current_checkpoint = self.info['checkpoint']
            current_seconds = self.info['clock_minutes']*60 + self.info['clock_seconds']

            if next_checkpoint > current_checkpoint:
                # passou um checkpoint
                reward += 5

            if next_lap > current_lap:
                # passou uma lap
                reward += 100
            

            # rank reward
            if next_rank > current_rank:
                reward += -2*(next_rank - current_rank)

            # promover velocidade
            if next_info['speed_south']**2 + next_info['speed_east']**2 == 0:
                reward -= 0.1


            terminated = (next_info['current_lap'] == 133) or (next_info['racers_finished'] == 14) # 14 é 7*2 ou seja se todos já acabaram
            
            # time penalty at the end of episode
            if terminated:
                reward -= -0.1*next_seconds


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


env = SuperMarioKartEnv()
# env = SuperMarioKartEnv(render_mode="human")

obs, info = env.reset()
terminated = False
while not terminated:
    next_observation, reward, terminated, _, next_info =  env.step([1,0,0,0,1,0,0,0,0])
    # env.render()



import matplotlib.pyplot as plt

history_east = next_info['position_history_east']
history_south = next_info['position_history_south']

plt.figure(figsize=(8, 8))
plt.plot(history_east, history_south, color='blue', alpha=0.5, label='Path')
plt.scatter(history_east, history_south, color='red', s=5, alpha=0.3, label='Positions')
plt.xlim(0, 4500)
plt.ylim(0, 4500)

ax = plt.gca()
ax.invert_yaxis()

# Labels and styling
plt.title('Agent Trajectory on Track')
plt.xlabel('Position East')
plt.ylabel('Position South')
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend()

plt.show()

# next_info['position_history'] = np.array((), dtype=np.int16)
# next_info['position_history'] = np.append(next_info['position_history'], 100)
# print(obs)
# print("---")
# print(info)