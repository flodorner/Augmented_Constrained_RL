#Install safety gym (requires Mujoco-py: https://github.com/openai/mujoco-py), openai gym, spinup (https://spinningup.openai.com/en/latest/) and pytorch first

import safety_gym
import gym
from spinup import sac_pytorch #sac_tf1 should work with tensorflow 1, if you prefer to use that instead...

env = gym.make('Safexp-PointGoal1-v0')

sac_pytorch(lambda: env,epochs=1000,alpha=0.2)
