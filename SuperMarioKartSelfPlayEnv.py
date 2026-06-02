from collections import deque

import torch
import stable_retro as retro
import stable_retro.data as stable_retro_data
import cv2
import numpy as np
import gymnasium as gym
from SuperMarioKartEnv import SuperMarioKartEnv
from stable_baselines3 import PPO
import random
import pdb

class SuperMarioKartSelfPlayEnv(SuperMarioKartEnv):
    def __init__(self, models_pool, frame_stack, frameskipN, render_mode="rgb_array"):
        # para ele treinar o modelo em ambas as personagens
        if random.sample([0,1], 1)[0] > 0.5:
            state = "mushroom_cup_100cc_2player_start.state"
        else:
            state = "mushroom_cup_100cc_2player_start_inverted_characters.state"
        

        super().__init__(frameskipN=frameskipN, game="/home/farrucho/Desktop/extra/mario-kart-rl/custom_integrations/SuperMarioKart-Snes", state=state, inttype=stable_retro_data.Integrations.ALL, render_mode=render_mode, players=2)
        self.models_pool = models_pool
        self.opponent_model = PPO.load(
            random.sample(self.models_pool, 1)[0],
            device="cuda",
        )
        self.frame_stack = frame_stack
        self.stacked_obs = deque(maxlen=self.frame_stack)

    def preprocessP1Obs(self, observation):
        img = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) # I CHANGED THIS WATCHOUT
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # I CHANGED THIS WATCHOUT
        img = img[3:107,0:257] # top part of screen (104, 256)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        
        # return img[np.newaxis, :, :].astype(np.float32) / 255.0
        return img[np.newaxis, :, :].astype(np.uint8)

    def preprocessP2Obs(self, observation):
        img = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) # I CHANGED THIS WATCHOUT
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # I CHANGED THIS WATCHOUT
        img = img[115:218,0:257] # top part of screen (104, 256)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        
        # return img[np.newaxis, :, :].astype(np.float32) / 255.0
        return img[np.newaxis, :, :].astype(np.uint8)

    def step(self, action):
        allowed_action = self.allowed_actions[action]

        full_actions = np.zeros(24, dtype=np.uint8)
        full_actions[self.mapped_indices] = allowed_action
        
        stacked_obs = np.concatenate(
            list(self.stacked_obs),
            axis=0
        )
        # print(stacked_obs)
        # print(stacked_obs.shape)
        # print(stacked_obs.dtype)
        # print(stacked_obs[0])

        # self_model_action, _states = self.opponent_model.predict(stacked_obs, deterministic=True)
        obs_tensor = torch.as_tensor(
            stacked_obs[None],
            device="cuda",
            dtype=torch.float32
        )

        with torch.no_grad():
            actions, _, _ = self.opponent_model.policy.forward(obs_tensor)

        self_model_action = int(actions.item())
        del obs_tensor, actions
        # self_model_action, _states = self.opponent_model.predict(stacked_obs)


        opponent_allowed_action = self.allowed_actions[self_model_action]


        full_actions[self.mapped_indices+12] = opponent_allowed_action


        reward = 0
        for j in range(0, self.frameskipN):
            next_observation_original, next_reward, next_terminated, _, next_info = retro.RetroEnv.step(self,full_actions)

            next_observation = self.preprocessP1Obs(next_observation_original)

            self.stacked_obs.append(self.preprocessP2Obs(next_observation_original).copy())
            if next_terminated:
                break
        

        if self.info is None or self.info == {}:
            # print("TRIGGER, FIRST STEP")
            self.pos_south_hist = [next_info['pos_south']]
            self.pos_east_hist = [next_info['pos_east']]
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
            next_info['reward_start_speed'] = 0


            self.observation, self.reward, self.terminated, self.info, = next_observation, next_reward, False, next_info        


            return next_observation, reward, next_terminated, False, next_info
        else:
            # store info
            self.pos_south_hist.append(next_info['pos_south'])
            self.pos_east_hist.append(next_info['pos_east'])
            
            next_info['last_checkpoint_time'] = self.info['last_checkpoint_time']
            next_info['checkpoints crossed'] = self.info['checkpoints crossed']

            next_info['reward_checkpoint'] = self.info['reward_checkpoint']
            next_info['reward_lap'] = self.info['reward_lap']
            next_info['reward_collision'] = self.info['reward_collision']
            next_info['reward_rank'] = self.info['reward_rank']
            next_info['reward_speed'] = self.info['reward_speed']
            next_info['reward_finish_race'] = self.info['reward_finish_race']
            next_info['reward_time'] = self.info['reward_time']
            next_info['reward_start_speed'] = self.info['reward_start_speed']


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
                    reward += max(1,3-(next_seconds - next_info['last_checkpoint_time']))
                    next_info['checkpoints crossed'][next_lap].add(next_checkpoint)
                    next_info['reward_checkpoint'] += max(1, 3-(next_seconds - next_info['last_checkpoint_time']))
                    next_info['last_checkpoint_time'] = next_seconds

            if next_lap > current_lap and next_lap != current_max_reached_lap and next_lap != 1: # check se realmente passou para uma nova lap e nao da uma volta inicial assim que a corrida comeca
                # passou uma lap
                reward += 20
                next_info['reward_lap'] += 20

            
            if next_collision == 7:
                # penalizar colisoes
                reward += -1
                next_info['reward_collision'] += -1
                # print("colisao")

            # rank reward per step
            # reward += (0.4*1/next_rank - 0.4*1/8) - 20*(next_rank - current_rank)
            # next_info['reward_rank'] += (0.4*1/next_rank - 0.4*1/8) - 20*(next_rank - current_rank)
            reward += 0.1*(1/next_rank - 1/8)
            next_info['reward_rank'] += 0.1*(1/next_rank - 1/8)
            

            # promover velocidade, encorajar para aprender mais cedo a nao ficar parado
            # module de cada componente da velocidade vai de de 0 até 1000, 2000 no maximo se usar cogumelo
            reward += 0.0001*(np.sqrt(next_info['speed_south']**2 + next_info['speed_east']**2)) # reward negativa se andar devagar
            next_info['reward_speed'] += 0.0001*np.sqrt(next_info['speed_south']**2 + next_info['speed_east']**2)
            # reward finishing race
            if next_lap == 6:
                reward += 50/next_rank
                next_info['reward_finish_race'] += 50/next_rank

            # reward aggresive start speed
            if next_seconds < 2:
                reward += 0.005*(np.sqrt(next_info['speed_south']**2 + next_info['speed_east']**2))
                next_info['reward_start_speed'] += 0.005*(np.sqrt(next_info['speed_south']**2 + next_info['speed_east']**2))

            # com frameskip de 4 obtém 1569 steps -> ~10steps/segundo
            # before 20M steps
            terminated = (next_lap == 6) or \
                            next_seconds - next_info['last_checkpoint_time'] > 8 or \
                            (next_seconds > 150) or \
                            (next_rank == 8 and next_seconds > 30)
            # after 20M steps
            # terminated = (next_lap == 6) or \
            #                 next_seconds - next_info['last_checkpoint_time'] > 8 or \
            #                 (next_seconds > 150)
            
            if terminated:
                next_info['position_history_south'] = np.array(self.pos_south_hist, dtype=np.int16)
                next_info['position_history_east'] = np.array(self.pos_east_hist, dtype=np.int16)
            else:
                next_info['position_history_south'] = np.array([], dtype=np.int16)
                next_info['position_history_east'] = np.array([], dtype=np.int16)
            # terminar se user acabou corrida ou nao houver progressao checkpoint ou episodio > 150s ou fez uma lap e ainda está em ultimo

        self.observation, self.reward, self.terminated, self.info = next_observation, reward, terminated, next_info
        return self.observation, self.reward, self.terminated, _, self.info
    
    def reset(self, seed=None, options=None):
        observation, info = retro.RetroEnv.reset(self,seed=seed, options=options)
        preprocessed_obs = self.preprocessP1Obs(observation)

        self.stacked_obs.clear()
        for _ in range(self.frame_stack):
            self.stacked_obs.append(self.preprocessP2Obs(observation).copy())

        self.pos_south_hist = []
        self.pos_east_hist = []
        self.observation = preprocessed_obs
        self.info = info
        self.reward = 0
        self.terminated = False

        return preprocessed_obs, info

# from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack, VecNormalize, VecMonitor
# import time

# if __name__ == "__main__":
#     num_cpu = 1
#     vec_env = SubprocVecEnv([lambda: SuperMarioKartSelfPlayEnv(models_pool=["/home/farrucho/Desktop/extra/mario-kart-rl/models/16yxsuaq/model.zip","/home/farrucho/Desktop/extra/mario-kart-rl/models/16yxsuaq/model_13M.zip","/home/farrucho/Desktop/extra/mario-kart-rl/models/3eacin6z/model_15M.zip","/home/farrucho/Desktop/extra/mario-kart-rl/models/3eacin6z/model_19M.zip"], frameskipN=4, frame_stack=4, render_mode="human") for _ in range(num_cpu)], start_method="spawn")
    
#     vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")

#     vec_env.reset()
#     while True:
#         actions = np.array([vec_env.action_space.sample() for _ in range(vec_env.num_envs)])

#         obs, rewards, dones , infos = vec_env.step(actions)
#         vec_env.render()

#         time.sleep(0.02)