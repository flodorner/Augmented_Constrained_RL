#Test if SAC is able to deal with the safety-gym environments when constraints are ignored.

#Install safety gym (requires Mujoco-py: https://github.com/openai/mujoco-py), openai gym, spinup (https://spinningup.openai.com/en/latest/) and pytorch first

import safety_gym
import gym
from spinup import sac_pytorch #sac_tf1 should work with tensorflow 1, if you prefer to use that instead...

env = gym.make('Safexp-PointGoal1-v0')
def provide_env():
    env.reset()
    return env 

sac_pytorch(provide_env,epochs=250,alpha=0.2)
