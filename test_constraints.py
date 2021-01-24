import safety_gym
import gym
from spinup import sac_pytorch,td3_pytorch,ppo_pytorch
import sys
from wrapper import constraint_wrapper
import pickle
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.test_policy import load_policy_and_env, run_policy
import matplotlib.pyplot as plt
from matplotlib import animation
import os
from datetime import datetime
import torch

def run_exp(alg="sac",alpha=None,add_penalty=1,mult_penalty=1,cost_penalty=0,buckets=None,
         epochs=30,start_steps=10000,ac_kwargs={"hidden_sizes":(256,256)},
            entropy_constraint=-1,filename="",steps_per_epoch=10001,
            act_noise=0.1,env_name='Safexp-PointGoal1-v0',batch_size=100,adaptive=False,adaptive_len=10,max_penalty=100):

    # alg determines whether sac, ppo or td3 is used.
    # alpha is the exploration parameter in sac. Add_parameter is Beta from the proposal.
    # If mult_penalty is not None, all rewards get multiplied by it once the constraint is violated (1-alpha from the proposal)
    # cost_penalty is equal to zeta from the proposal. buckets determines how the accumulated cost is discretized for the agent:
    #if it is None, cost is a continouos variable, else there are buckets indicator variables for a partition of [0,constraint]
    # (with the last only activating if the constraint is violated). Epochs indicates, how many epochs to train for, start_steps indicates
    # how many random exploratory actions to perform before using the trained policy. Split_policy changes the network architecture
    # such that a second network is used for the policy and q-values when the constraint is violated. ac_kwargs is a dict containing
    # the arguments for the actor-critic class. Hidden sizes is a tuple containing the sizes for all hidden layers.
    # safe_policy indicates the saving location for a trained safe policy. If provided, the safe policy will take over whenever the constraint
    # is violated. filename determines, where in the results folder the res29541.pts-3.tensorflow-1-vmults and trained policy get saved to.
    # Steps_per epoch determines the amount of environment interaction per epoch. Num_test episodes the amount of test_episodes
    # (only for evaluation) that are performed after each epoch. act_noise controls the exploration noise used in the td3 algorithm.
    # Entropy_constraint is the entropy to aim for (if sac is used with trainable alpha)
    if buckets == -1:
        buckets = None
    if entropy_constraint == 0:
        entropy_constraint = None
    if alpha == 0:
        alpha = None

    env = gym.make(env_name) # Create an instance of the safety-gym environment.
    # Create an instance of the constrained environment.
    env = constraint_wrapper(env,add_penalty=add_penalty,mult_penalty=mult_penalty,
                             cost_penalty=cost_penalty,buckets=buckets,adaptive=adaptive,adaptive_len=adaptive_len,max_penalty=max_penalty)
    logger_kwargs = setup_logger_kwargs(filename+"policy",data_dir="results/")
    assert alg == "sac" or alg == "td3" or alg == "ppo"
    # Select learning method
    if alg == "sac":
        import spinup.algos.pytorch.sac.core as core
        actor_critic = core.MLPActorCritic
        # Start training with SAC
        sac_pytorch(lambda: env,epochs=epochs,alpha=alpha,steps_per_epoch=steps_per_epoch,start_steps=start_steps,
                    logger_kwargs=logger_kwargs,actor_critic=actor_critic,ac_kwargs=ac_kwargs,entropy_constraint=entropy_constraint,batch_size=batch_size)
    elif alg == "td3":
        import spinup.algos.pytorch.td3.core as core
        actor_critic = core.MLPActorCritic
        # Start training with TD3
        td3_pytorch(lambda: env,epochs=epochs,steps_per_epoch=steps_per_epoch,start_steps=start_steps,logger_kwargs=logger_kwargs,
                    actor_critic=actor_critic,act_noise=act_noise,ac_kwargs=ac_kwargs,batch_size=batch_size)
    elif alg == "ppo":
        import spinup.algos.pytorch.ppo.core as core
        actor_critic = core.MLPActorCritic
        # Start training with PPO
        ppo_pytorch(lambda: env, epochs=epochs, steps_per_epoch=steps_per_epoch,
                    logger_kwargs=logger_kwargs,
                    actor_critic=actor_critic, ac_kwargs=ac_kwargs)

    #Ideally, you would separate train and test runs more directly here rather than reylying on the alg to work exactly as described...
    # Store training results in pickle file
    with open(filename+"rews.pkl", 'wb') as f:
        pickle.dump(env.total_rews, f)
    with open(filename+"costs.pkl", 'wb') as f:
        pickle.dump(env.total_costs, f)
    if env.adaptive:
        with open(filename + "pens.pkl", 'wb') as f:
            pickle.dump(env.adaptive, f)

