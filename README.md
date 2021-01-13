# Augmented Contstrained RL

Constrained Reinforcement Learning using State Augmentation

## Supported Platforms

This package has been tested on CentOS 7 and Ubuntu 20.04 LTS, and probably works fine for most recent Linux operating systems. 

Requires **Python 3.6.x**  

## Installation

This code depends on [mujoco-py](https://github.com/openai/mujoco-py), [safety-gym](https://github.com/openai/safety-gym) and [spinningup](https://github.com/openai/spinningup) by OpenAI. This sections runs you through a quick installation of the required python packages.

### Installing MuJoCo

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

### Installing Safety Gym

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

### Installing SpinningUp

[SpinningUp](https://spinningup.openai.com/en/latest/) contains a code repo of the implementation of key Reinforcement Learning algorithms including Soft Actor-Critic, Proximal Policy Optimization and Twin Delayed DDPG used in this project.

We use a [forked repository](https://github.com/flodorner/spinningup) of the original SpinninpUp where we implement changes required for State Augmented Constrained RL. 

1. First install OpenMPI:
```
sudo apt-get update && sudo apt-get install libopenmpi-dev
```
2. Now install the forked spinningup:
```
git clone https://github.com/flodorner/spinningup.git
cd spinningup
pip install -e .
```

## Code Structure

**wrapper.py:** Defines the ```constraint_wrapper``` class which serves as a wrapper around the safety-gym ```env``` class. The ```step``` method of the ```constraint_wrapper``` class returns a cost-augmented state observations and a cost-modified reward.

**test_constraints.py:** Creates an instance of the ```constraint_wrapper``` class and starts an experiment with given arguments.

**test_noconstraints.py:** Creates an instance of the safety-gym ```env``` class and starts training an unconstrained agent with default arguments.

**run.py:** Run an experiment from the set of experiments listed in the proposal.


## Running Experiments

### Predefined Experiments

The experiments can simply be run by:
```
python run.py --id {exp_id}
```
where _{exp_id}_ is the id of the experiment you wish to run.

### Custom Experiments

You can also run your custom experiments by passing runtime arguments to _test_constraints.py_. For example, start an experiment with Soft Actor-Critic by running:
```
python test_constraints.py --alg 'sac'
```
The following arguments are available:
```
--alg: alg determines wheter sac, ppo or td3 is used.
--alpha: alpha is the exploration parameter in sac.
--add_penalty: add_penalty is Beta from the proposal.
--mult_penalty: If mult_penalty is not None, all rewards get multiplied by it once the constraint is violated (1-alpha from the proposal).
--cost_penalty: cost_penalty is equal to zeta from the proposal.
--buckets: buckets determines how the accumulated cost is discretized for the agent.
--epochs: Epochs indicates how many epochs to train for.
--start_step: start_steps indicates how many random exploratory actions to perform before using the trained policy. 
--split_policy: split_policy changes the network architecture such that a second network is used for the policy and q-values when the constraint is violated. 
--safe_policy: safe_policy indicates the saving location for a trained safe policy. If provided, the safe policy will take over whenever the constraint is violated.
--name: name determines where in the results folder the res29541.pts-3.tensorflow-1-vmults and trained policy get saved to.
--env_name: env_name indicates the name of the enviroment the agent trains on. Can be chosen from one of the safety-gym environments.
```

## Results

The results of all the experiments listed in run.py can be found at this [polybox link](https://polybox.ethz.ch/index.php/s/ElsdfFGYtBiVq3L).

