# Super Mario Kart RL

Reinforcement Learning agent for **Super Mario Kart (SNES)** using:
* `stable-retro`
* `PPO`
* custom Gymnasium environment
* custom reward shaping
* frame stacking + CNN observations

The agent is able to:
* complete races consistently
* achieve top placements
* make smart shortcuts
* defense (dodge other players items)
* attack (use items at the right time)
* avoid collisions and offtrack

---

![Rollout Buffer during training](media/demo.gif)

---


# Main Features

* Custom `RetroEnv` wrapper with frameskipping
* Reduced discrete action space (this improves PPO training stability BY A LOT from 2^12=4096 to 18 possible combination of actions)
* PPO training (60M Steps) with parallel environments in cpu with inference on GPU RTX 3060
* Reward shaping for racing behavior based on RAM addresses
* Observation space (`4x84x84`) is preprocessed (cropped->grayscale->resized->frame stacking)
* Terminated conditions (no checkpoint progress, finish first lap at last place, and more) allow the PPO algorithm to converge much faster
* Checkpoint/lap tracking
* WandB integration
* Video recording during training
* Simple Nature CNN network

---

# Reward Function

The reward function combines several components:

* checkpoint progression
* lap completion
* race finish bonus
* speed reward
* rank shaping
* collision penalties



---

# Training Setup

hyperparameters used at start of training:

```python
learning_rate = 2.5e-4
n_steps = 1024
batch_size = 256
n_epochs = 5
gamma = 0.99
ent_coef = 0.01
```

Parallel training:

```python
12 environments
```
---

# Acknowledgments and Bibliography

* OpenAI Gym Retro
* Stable-Baselines3
* Nintendo / Super Mario Kart SNES
* https://datacrystal.tcrf.net/wiki/Super_Mario_Kart/RAM_map
* https://bin.smwcentral.net/u/34395/SMK_Potential_Ram_Addresses.txt
