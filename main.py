import stable_retro as retro
import stable_retro.data as stable_retro_data
import os
import cv2
import numpy as np

class SuperMarioKartEnv(retro.RetroEnv):
    def __init__(self):
        super().__init__(game="SuperMarioKart-Snes", inttype=stable_retro_data.Integrations.ALL)
    
    def step(self, *params):
        observation, reward, terminated, _, info = super().step(*params)
        
        terminated = (info['current_lap'] == 133) or (info['racers_finished'] == 14) # 14 é 7*2 ou seja se todos já acabaram

        return observation, reward, terminated, _, info

    def save_png_observation(self, observation, filepath="./images/test_observation.png"):
        cv2.imwrite(filepath, cv2.cvtColor(observation, cv2.COLOR_RGB2BGR))


# rew, done, info = self.compute_step()


# def main():
stable_retro_data.Integrations.add_custom_path("/home/farrucho/Desktop/extra/mario-kart-rl/custom_integrations")
print("SuperMarioKart-Snes" in stable_retro_data.list_games(inttype=stable_retro_data.Integrations.ALL))

env = SuperMarioKartEnv()
print(env.action_space) # 12 buttons

env.reset()
action = env.action_space.sample()
observation, reward, terminated, _, info = env.step(action)
print(observation.shape)
env.save_png_observation(observation)
# print(reward)
# print(terminated)
# print(info)


# while True:
#     action = env.action_space.sample()
#     observation, reward, terminated, _, info = env.step(action)
#     print(info)
#     env.render()
#     if terminated:
#         env.reset()
env.close()



# import retro
# import os

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# def main():
#         retro.data.Integrations.add_custom_path(
#                 os.path.join(SCRIPT_DIR, "custom_integrations")
#         )
#         print("FakeGame-Nes" in retro.data.list_games(inttype=retro.data.Integrations.ALL))
#         env = retro.make("FakeGame-Nes", inttype=retro.data.Integrations.ALL)
#         print(env)

