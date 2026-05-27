import stable_retro as retro
import stable_retro.data as stable_retro_data
import cv2
import numpy as np
import gymnasium as gym


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
        
        # the space was too big, 2**9=512 so we must reduce even more
        self.allowed_actions = np.array([
            # Base Driving
            [0, 0, 0,  0,  0,    0,     0, 0, 0],  # 0: NOOP
            [1, 0, 0,  0,  0,    0,     0, 0, 0],  # 1: Accelerate
            [1, 0, 0,  0,  1,    0,     0, 0, 0],  # 2: Accel + Steer Left
            [1, 0, 0,  0,  0,    1,     0, 0, 0],  # 3: Accel + Steer Right
            [0, 1, 0,  0,  0,    0,     0, 0, 0],  # 4: Brake
            [0, 1, 0,  0,  1,    0,     0, 0, 0],  # 5: Brake + Steer Left
            [0, 1, 0,  0,  0,    1,     0, 0, 0],  # 6: Brake + Steer Right
            [0, 0, 0,  0,  1,    0,     0, 0, 0],  # 7: Coast + Steer Left
            [0, 0, 0,  0,  0,    1,     0, 0, 0],  # 8: Coast + Steer Right

            # Advanced Movement: Power Sliding and Hopping
            [1, 0, 0,  0,  0,    0,     0, 0, 1],  # 9:  Accel + Hop (Straight jump)
            [1, 0, 0,  0,  1,    0,     0, 0, 1],  # 10: Accel + Power Slide Left
            [1, 0, 0,  0,  0,    1,     0, 0, 1],  # 11: Accel + Power Slide Right

            # Wall Recovery / Standstill Turns
            [0, 0, 0,  0,  1,    0,     0, 0, 1],  # 12: Hop + Steer Left (Pivot when stuck)
            [0, 0, 0,  0,  0,    1,     0, 0, 1],  # 13: Hop + Steer Right (Pivot when stuck)

            # Item Usage
            [1, 0, 0,  0,  0,    0,     1, 0, 0],  # 14: Accel + Item
            [1, 0, 0,  0,  1,    0,     1, 0, 0],  # 15: Accel + Steer Left + Item
            [1, 0, 0,  0,  0,    1,     1, 0, 0],  # 16: Accel + Steer Right + Item

            # Advanced Item Usage
            [1, 0, 0,  1,  0,    0,     1, 0, 0],  # 17: Accel + Down + Item (Drop backward)
            [1, 0, 1,  0,  0,    0,     1, 0, 0],  # 18: Accel + Up + Item (Throw forward)

            # (Sliding + Item Usage)
            [1, 0, 0,  0,  1,    0,     1, 0, 1],  # 19: Accel + Slide Left + Item
            [1, 0, 0,  0,  0,    1,     1, 0, 1],  # 20: Accel + Slide Right + Item
        ], dtype=np.uint8)

        self.action_space = gym.spaces.Discrete(len(self.allowed_actions), dtype=np.uint8)
        # now the action space goes from 0 to len(self.allowed_actions)

        self.frameskipN = frameskipN
        self.lastPos = None

    def preprocessObs(self, observation):
        img = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY) # I CHANGED THIS WATCHOUT
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # I CHANGED THIS WATCHOUT
        img = img[3:107,0:257] # top part of screen (104, 256)
        img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        
        return img[np.newaxis, :, :].astype(np.float32) / 255.0

    def step(self, action):
        allowed_action = self.allowed_actions[action]

        full_actions = np.zeros(12, dtype=np.uint8)
        full_actions[self.mapped_indices] = allowed_action

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
            next_info['reward_start_speed'] = 0


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
                    reward += 4-(next_seconds - next_info['last_checkpoint_time'])
                    next_info['checkpoints crossed'][next_lap].add(next_checkpoint)
                    next_info['reward_checkpoint'] += 2-(next_seconds - next_info['last_checkpoint_time'])
                    next_info['last_checkpoint_time'] = next_seconds

            if next_lap > current_lap and next_lap != current_max_reached_lap and next_lap != 1: # check se realmente passou para uma nova lap e nao da uma volta inicial assim que a corrida comeca
                # passou uma lap
                reward += 100
                next_info['reward_lap'] += 100

            
            if next_collision == 7:
                # penalizar colisoes
                reward += -1
                next_info['reward_collision'] += -1
                # print("colisao")

            # rank reward per step
            # reward += (0.4*1/next_rank - 0.4*1/8) - 20*(next_rank - current_rank)
            # next_info['reward_rank'] += (0.4*1/next_rank - 0.4*1/8) - 20*(next_rank - current_rank)
            reward += 5*(1/next_rank - 1/8)
            next_info['reward_rank'] += 5*(1/next_rank - 1/8)
            

            # promover velocidade, encorajar para aprender mais cedo a nao ficar parado
            # module de cada componente da velocidade vai de de 0 até 1000, 2000 no maximo se usar cogumelo
            reward += 0.0001*(np.sqrt(next_info['speed_south']**2 + next_info['speed_east']**2)) # reward negativa se andar devagar
            next_info['reward_speed'] += 0.0001*np.sqrt(next_info['speed_south']**2 + next_info['speed_east']**2)
            # reward finishing race
            if next_lap == 6:
                reward += 300/next_rank
                next_info['reward_finish_race'] += 300/next_rank

            # reward aggresive start speed
            if next_seconds < 5:
                reward += 0.0015*(np.sqrt(next_info['speed_south']**2 + next_info['speed_east']**2))
                next_info['reward_start_speed'] += 0.002*(np.sqrt(next_info['speed_south']**2 + next_info['speed_east']**2))

            # com frameskip de 4 obtém 1569 steps -> ~10steps/segundo
            terminated = (next_lap == 6) or \
                            next_seconds - next_info['last_checkpoint_time'] > 8 or \
                            (next_seconds > 150) or \
                            (next_rank == 8 and next_seconds > 20)
            # terminar se user acabou corrida ou nao houver progressao checkpoint ou episodio > 150s ou fez uma lap e ainda está em ultimo
            
            # time penalty at the end of race
            # if next_lap == 6:
            #     # print(next_seconds) # at max it will be 94
            #     reward += -next_seconds
            #     next_info['reward_time'] += -next_seconds
            print(current_rank, next_rank)

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
        cv2.imwrite(filepath, (observation[0] * 255).astype(np.uint8))
        # img = cv2.cvtColor(observation, cv2.COLOR_RGB2BGR)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = img[3:107,0:257] # top part of screen (104, 256)
        # img = cv2.resize(img, (84, 84), interpolation=cv2.INTER_AREA)
        # cv2.imwrite(filepath, img)
