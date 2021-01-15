import gym
from gym.spaces import Box
import numpy as np
from spinup.utils.test_policy import load_policy_and_env
import torch

# Used to represent accumulated cost in the form of discrete buckets. All buckets below the cost value are set to 1.
def bucketize(x,n_buckets,max_x):
    out = np.zeros(n_buckets)
    for i in range(1,n_buckets+1):
        if x>i*max_x/n_buckets:
            out[i-1]=1
    return out

# Wrapper around the safety-gym env class
class constraint_wrapper:
    def __init__(self, env,add_penalty=10,threshold=25,mult_penalty=None,cost_penalty=0,
                 buckets=None,safe_policy=False):
        self.base_env = env 
        self.buckets = buckets # no. of buckets for discretization
        # Adding cost dimension to observation space
        if self.buckets is None: 
            low = np.concatenate([env.observation_space.low,np.array([0])])
            high = np.concatenate([env.observation_space.high,np.array([np.inf])])
        else:
            low = np.concatenate([env.observation_space.low,np.array([0 for i in range(self.buckets)])])
            high = np.concatenate([env.observation_space.high,np.array([np.inf for i in range(self.buckets)])])
        self.observation_space = Box(low=low,high=high,dtype=np.float32) 
        self.action_space = env.action_space
        self.total_rews = [] # To store total episode returns
        self.total_costs = [] # To store total episode costs
        self.t = -1
        self.add_penalty = add_penalty # add_penalty is Beta from the proposal.
        self.threshold = threshold # threshold value for cost
        self.mult_penalty = mult_penalty # If mult_penalty is not None, all rewards get multiplied by it once the constraint is violated.
        self.cost_penalty = cost_penalty # cost_penalty is equal to zeta from the proposal.
        # use a safe_policy if constraint voilated
        if safe_policy is not False:
            _,self.safe_policy = load_policy_and_env(safe_policy)
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.safe_policy = False

    def reset(self):
        self.penalty_given = False # Reset penalties
        if self.t > 0:
            self.total_rews.append(self.reward_counter)
            self.total_costs.append(self.cost_counter)
        obs = self.base_env.reset()
        # Reset episode time, cost, and returns
        self.t = 0
        self.cost_counter = 0
        self.reward_counter = 0
        if self.buckets is None:
            self.obs_old = np.concatenate([obs, [self.cost_counter]])
        else:
            self.obs_old = np.concatenate([obs, bucketize(self.cost_counter,self.buckets,self.threshold)])
        return self.obs_old

    def step(self,action):
        if self.base_env.done:
            self.base_env.reset()
        # Use safe policy if asked
        if self.safe_policy is not False:
            if self.cost_counter >= self.threshold:
                action = self.safe_policy(torch.tensor(self.obs_old).to(self.device))
        obs, reward, done, info = self.base_env.step(action)
        self.reward_counter += reward # Update total episode reward
        self.cost_counter += info["cost"] # Update total episode cost
        self.t += 1
        # Calculate the cost adjusted reward
        if self.mult_penalty is not None:
            if self.cost_counter>self.threshold:
                reward = reward * self.mult_penalty
        r_mod = reward-self.add_penalty*(self.cost_counter>self.threshold)*(not self.penalty_given)
        r_mod = r_mod - (self.cost_counter>self.threshold)*info["cost"]*self.cost_penalty
        self.penalty_given = self.cost_counter>self.threshold
        # Augment observation space with accumulated cost
        if self.buckets is None:
            self.obs_old = np.concatenate([obs, [min(self.cost_counter,self.threshold+1)]])
        else:
            self.obs_old = np.concatenate([obs,bucketize(self.cost_counter,self.buckets,self.threshold)])
        return self.obs_old, r_mod, done, info
    def render(self, mode='human'):
        return self.base_env.render(mode,camera_id=1)


# Wrapper class for Safety-Gym
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






