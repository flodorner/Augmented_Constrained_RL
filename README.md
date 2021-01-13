# Augmented Contstrained RL

Constrained Reinforcement Learning using State Augmentation

## Supported Platforms

This package has been tested on CentOS 7 and Ubuntu 20.04 LTS, and probably works fine for most recent Linux operating systems. 

Requires **Python 3.6.x**  

## Installation

This code depends on [mujoco_py](https://github.com/openai/mujoco-py), [safety_gym](https://github.com/openai/safety-gym) and [spinningup](https://github.com/openai/spinningup). This sections runs you through a quick installation of the required python packages.

#### Installing MuJoCo

MuJoCo is a physics engine for detailed, efficient rigid body simulations with contacts. mujoco-py allows using MuJoCo from Python 3. Further details about mujoco-py can be found [here](https://github.com/openai/mujoco-py).

1. Obtain a MuJoCo trial license by visiting the [Mujoco website](https://www.roboti.us/license.html). You can also request for a student license. 
2. Download the mujoco200 package from this [link](https://www.roboti.us/download/mujoco200_linux.zip).
3. Unzip the downloaded mujoco200 directory into ~/.mujoco/mujoco200, and place your license key (the mjkey.txt file from your license email) at ~/.mujoco/mjkey.txt.
4. Before installing mujoco-py on Ubuntu, make sure you have the following libraries installed:
```
sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3
```
5. Now install mujoco-py using pip:
```
pip install -U 'mujoco-py<2.1,>=2.0'
```

#### Installing Safety Gym

Safety Gym is a suite of environments and tools for measuring progress towards reinforcement learning agents that respect safety constraints while training. More information can be found [here](https://openai.com/blog/safety-gym/)

1. First install openai-gym:
```
pip install gym
```
2. Afterwards, simply install Safety Gym by:
```
git clone https://github.com/openai/safety-gym.git

cd safety-gym

pip install -e .
```




Use the following fork for spinup:
https://github.com/flodorner/spinningup
