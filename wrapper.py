import gym
from gym.spaces import Box
import numpy as np



class constraint_wrapper:
    def __init__(self, env,penalty=10,threshold=25):
        self.base_env = env
        low = np.concatenate([env.observation_space.low,np.array([0])])
        high = np.concatenate([env.observation_space.high,np.array([np.inf])])
        self.observation_space = Box(low=low,high=high,dtype=np.float32)
        self.action_space = env.action_space
        self.total_rews = []
        self.total_costs = []
        self.t = -1
        self.penalty = penalty
        self.threshold = threshold

    def reset(self):
        if self.t > 0:
            self.total_rews.append(self.reward_counter)
            self.total_costs.append(self.cost_counter)
        obs = self.base_env.reset()
        self.t = 0
        self.cost_counter = 0
        self.reward_counter = 0
        return np.concatenate([obs, [self.cost_counter]])
    def step(self,action):
        if self.base_env.done:
            self.base_env.reset()
        obs, reward, done, info = self.base_env.step(action)
        self.reward_counter += reward
        self.cost_counter += info["cost"]
        self.t += 1
        return np.concatenate([obs, [self.cost_counter]]), reward-self.penalty*(self.cost_counter>self.threshold) ,done, None


class safetygymwrapper:
    def __init__(self, env):
        self.base_env = env
    def reset(self):
        return self.base_env.reset()
    def step(self):
        obs, reward, done, info = self.base_env.step(action)
        cost = info["cost"] 
        return obs, np.array([reward,cost]),done,None

class fwrapper:
    def __init__(self,env,f,gamma=[0.99,1],fdims=2,gamma_learner = 1):
        self.base_env = env
        self.fdims = fdims
        self.f = f
        low = np.concatenate([env.observation_space.low,np.array([0]+[-np.inf for i in self.fdims])])
        high = np.concatenate([env.observation_space.high,np.array([np.inf]+[np.inf for i in self.fdims])])
        self.observation_space = Box(low=low,high=high,dtype=np.float32)
        self.action_space = env.action_space
        #
        if type(gamma) == float or type(gamma) == int:
            self.gamma=np.array([gamma for i in range(self.fdims)])
        else:
            self.gamma = np.array(gamma)
            assert len(gamma) == self.fdims
        self.gamma_learner = gamma_learner
        self.t = -1
        self.episode_rewards_disc = []
        self.episode_rewards_nodisc = []
    def reset(self):
        if self.t>0:
            self.episode_rewards_disc.append(self.r_disc)
            self.episode_rewards_nodisc.append(self.r_nodisc)
        obs = self.base_env.reset()
        self.t = 0
        self.r_disc = np.array([0 for i in range(self.fdims)])
        self.r_nodisc = np.array([0 for i in range(self.fdims)])
        return np.concatenate([obs,np.array([self.t]),self.r_disc])
    def step(self,action):
        obs, rewards, done, _ = self.base_env.step(action)
        f_old = self.f(rewards)
        self.r_disc += (self.gamma**self.t)*rewards
        self.r_nodisc += rewards
        f_rew = (self.f(rewards)-f_old)/(self.gamma_learner**self.t)
        self.t += 1
        return np.concatenate([obs, np.array([self.t]), self.r_disc]), f_rew ,done, None






