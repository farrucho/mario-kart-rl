# objective:
# load the model
# make it play 100 games in each new map and see performance


from SuperMarioKartEnv import SuperMarioKartEnv
from SuperMarioKartSelfPlayEnv import SuperMarioKartSelfPlayEnv
from NatureCNN import NatureCNN
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.callbacks import CheckpointCallback
import numpy as np
import json


num_cpu = 16
frameskip = 4
framestack = 4

infos = {} # dictionary with state objects, each objetc has array with infos
states = [
            "bowsercastle1_100cc_1player_start.state",
            "mariocircuit1_100cc_1player_start.state",
            "mariocircuit2_100cc_1player_start.state",
            "ghostvalley1_100cc_1player_start.state",
            "donutplains1_100cc_1player_start.state",
            #----
            "chocoisland1_100cc_1player_start.state",
            "ghostvalley2_100cc_1player_start.state",
            "donutplains2_100cc_1player_start.state",
            "bowsercastle2_100cc_1player_start.state",
            "mariocircuit3_100cc_1player_start.state",
            # --
            "koopabeach1_100cc_1player_start.state",
            "chocoisland2_100cc_1player_start.state",
            "vanillalake1_100cc_1player_start.state",
            "bowsercastle3_100cc_1player_start.state",
            "mariocircuit4_100cc_1player_start.state",
        ]

episodes = 20 # number of episodes per track to simulate

for state in states:
    vec_env = SubprocVecEnv([lambda cpu_n = cpu_n: Monitor(SuperMarioKartEnv(state=state, frameskipN=frameskip)) for cpu_n in range(num_cpu)], start_method="fork")
    vec_env = VecFrameStack(vec_env, n_stack=framestack, channels_order="first")

    model = PPO.load(
                "models/6axs4z55/checkpoints/model_204400000_steps.zip",
                device="cuda"
            )
    current_infos = []

    obs = vec_env.reset()
    while len(current_infos) < episodes:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = vec_env.step(action)
        
        indices = np.argwhere(dones).flatten()
        for done_indice in indices:
            if len(current_infos) < episodes: 
                current_info = {}
                current_info["current_lap"] = info[done_indice]["current_lap"]-127
                current_info["clock_final_time"] = info[done_indice]["clock_minutes"]*60+info[done_indice]["clock_seconds"]
                current_info["rank"] = info[done_indice]["rank"]/2+1
                current_info["reward_checkpoint"] = info[done_indice]["reward_checkpoint"]
                current_info["reward_collision"] = info[done_indice]["reward_collision"]
                current_info["reward_rank"] = info[done_indice]["reward_rank"]
                current_info["reward_speed"] = info[done_indice]["reward_speed"]
                current_info["reward_finish_race"] = info[done_indice]["reward_finish_race"]
                current_infos.append(current_info)

        infos[state] = current_infos
        # vec_env.render("human")
        
        # print("-----info----")
        # print(info)
        # print("-----dones----")
        # print(dones)
        # print("-----rewards----")
        # print(rewards)
        # vec_env.close()
       # infos[state] = infos[state].append(current_info)
    print(f"grabbed runs for state {state}")
print(infos)
with open('data/runs_info.json','w') as file:
    file.write(json.dumps(infos, indent=4))
