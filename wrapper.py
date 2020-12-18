import gym
from gym.spaces import Box
import numpy as np

def bucketize(x,n_buckets,max_x):
    out = np.zeros(n_buckets)
    for i in range(1,n_buckets+1):
        if x>i*max_x/n_buckets:
            out[i-1]=1
    return out

class constraint_wrapper:
    def __init__(self, env,add_penalty=10,threshold=25,keep_add_penalty=True,mult_penalty=None,cost_penalty=None,buckets=None):
        self.base_env = env
        self.buckets = buckets
        if self.buckets is None:
            low = np.concatenate([env.observation_space.low,np.array([0])])
            high = np.concatenate([env.observation_space.high,np.array([np.inf])])
        else:
            low = np.concatenate([env.observation_space.low,np.array([0 for i in range(self.buckets)])])
            high = np.concatenate([env.observation_space.high,np.array([np.inf for i in range(self.buckets)])])
        self.observation_space = Box(low=low,high=high,dtype=np.float32)
        self.action_space = env.action_space
        self.total_rews = []
        self.total_costs = []
        self.t = -1
        self.add_penalty = add_penalty
        self.threshold = threshold
        self.keep_add_penalty = keep_add_penalty
        self.mult_penalty = mult_penalty
        self.cost_penalty = cost_penalty
    def reset(self):
        self.penalty_given = False
        if self.t > 0:
            self.total_rews.append(self.reward_counter)
            self.total_costs.append(self.cost_counter)
        obs = self.base_env.reset()
        self.t = 0
        self.cost_counter = 0
        self.reward_counter = 0
        if self.buckets is None:
            return np.concatenate([obs, [self.cost_counter]])
        else:
            return np.concatenate([obs, bucketize(self.cost_counter,self.buckets,self.threshold)])
    def step(self,action):
        if self.base_env.done:
            self.base_env.reset()
        obs, reward, done, info = self.base_env.step(action)
        self.reward_counter += reward
        self.cost_counter += info["cost"]
        self.t += 1
        if self.mult_penalty is not None:
            if self.cost_counter>self.threshold:
                reward = reward * self.mult_penalty
        r_mod = reward-self.add_penalty*(self.cost_counter>self.threshold)*(self.keep_add_penalty or not self.penalty_given)
        if self.cost_penalty is not None:
            r_mod = r_mod - (self.cost_counter>self.threshold)*info["cost"]*self.cost_penalty
        if not self.keep_add_penalty:
            self.penalty_given = self.cost_counter>self.threshold
        if self.buckets is None:
            return np.concatenate([obs, [min(self.cost_counter,self.threshold+1)]]), r_mod,done, None
        else:
            return np.concatenate([obs,bucketize(self.cost_counter,self.buckets,self.threshold)]), r_mod, done, None


class safetygymwrapper:
    def __init__(self, env):
        self.base_env = env
    def reset(self):
        return self.base_env.reset()
    def step(self,action):
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






