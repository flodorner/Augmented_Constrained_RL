import safety_gym
import gym
from spinup import sac_pytorch,td3_pytorch #sac_tf1 should work with tensorflow 1, if you prefer to use that instead
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

def run_exp(alg="sac",alpha=0.02,add_penalty=1,keep_add_penalty=True,mult_penalty=None,cost_penalty=None,buckets=None,
         epochs=30,start_steps=10000,cost_penalty_always=False,split_policy=False,ac_kwargs={"hidden_sizes":(256,256)},
            use_safe_policy=False,filename="",steps_per_epoch=10001,num_test_episodes=10,act_noise=0.1):
    # alg determines wheter sac or td3 is used.
    #alpha is the exploration parameter in sac. Add_parameter is Beta from the proposal. If keep_add_penalty is true,
    # beta is subtracted from the reward at each step after the constraint is violated, not just the first time.
    # If mult_penalty is not None, all rewards get multiplied by it once the constraint is violated (1-alpha from the proposal)
    # cost_penalty is equal to zeta from the proposal. buckets determines how the accumulated cost is discretized for the agent:
    #if it is None, cost is a continouos variable, else there are buckets indicator variables for a partition of [0,constraint]
    # (with the last only activating if the constraint is violated). Epochs indicates, how many epochs to train for, start_steps indicates
    # how many random exploratory actions to perform before using the trained policy. If cost_penalty_always is True,
    # the penalty for incurring costs is also applied before the constraint is violated. Split_policy changes the network architecture
    # such that a second network is used for the policy and q-values when the constraint is violated. ac_kwargs is a dict containing
    # the arguments for the actor-critic class. Hidden sizes is a tuple containing the sizes for all hidden layers.
    # safe_policy indicates the saving location for a trained safe policy. If provided, the safe policy will take over whenever the constraint
    # is violated. filename determines, where in the results folder the results and trained policy get saved to.
    # Steps_per epoch determines the amount of environment interaction per epoch. Num_test episodes the amount of test_episodes
    # (only for evaluation) that are performed after each epoch. act_noise controls the exploration noise used in the td3 algorithm.
    #
    if mult_penalty == -1:
        mult_penalty = None
    if cost_penalty == -1:
        cost_penalty = None
    if buckets == -1:
        buckets = None

    env = gym.make('Safexp-PointGoal1-v0')
    env = constraint_wrapper(env,add_penalty=add_penalty,keep_add_penalty=keep_add_penalty,mult_penalty=mult_penalty,
                             cost_penalty=cost_penalty,buckets=buckets,cost_penalty_always=cost_penalty_always,safe_policy=safe_policy)
    logger_kwargs = setup_logger_kwargs(filename+"policy",data_dir="results/")
    assert alg == "sac" or alg == "td3"
    if alg == "sac":
        import spinup.algos.pytorch.sac.core as core
        if split_policy:
            actor_critic = core.MLPActorCriticSplit
        else:
            actor_critic = core.MLPActorCritic
        sac_pytorch(lambda: env,epochs=epochs,alpha=alpha,steps_per_epoch=steps_per_epoch,start_steps=start_steps,
                    logger_kwargs=logger_kwargs,num_test_episodes=num_test_episodes,actor_critic=actor_critic,ac_kwargs=ac_kwargs)
    elif alg == "td3":
        import spinup.algos.pytorch.td3.core as core
        if split_policy:
            actor_critic = core.MLPActorCriticSplit
        else:
            actor_critic = core.MLPActorCritic
        td3_pytorch(lambda: env,epochs=epochs,steps_per_epoch=steps_per_epoch,start_steps=start_steps,logger_kwargs=logger_kwargs,
                    actor_critic=actor_critic,num_test_episodes=num_test_episodes,act_noise=act_noise,ac_kwargs=ac_kwargs)

    #Ideally, you would separate train and test runs more directly here rather than reylying on the alg to work exactly as described...
    with open("results/"+filename+"rews.pkl", 'wb') as f:
        pickle.dump(env.total_rews, f)
    with open("results/"+filename+"costs.pkl", 'wb') as f:
        pickle.dump(env.total_costs, f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    _, get_action = load_policy_and_env("results/"+filename+"policy",deterministic=True)
    frames = []

    for i in range(5):
        o = env.reset()
        for i in range(1000):
            frames.append(env.render(mode="rgb_array"))
            a = get_action(torch.tensor(o).to(device))
            o, r, d, _ = env.step(a)
    if alg=="sac":
        _, get_action = load_policy_and_env("results/" + filename + "policy", deterministic=False)
        for i in range(5):
            o = env.reset()
            for i in range(1000):
                frames.append(env.render(mode="rgb_array"))
                a = get_action(torch.tensor(o).to(device))
                o, r, d, _ = env.step(a)

        
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    plt.axis('off')
    patch = plt.imshow(frames[0])
    gif = animation.FuncAnimation(plt.gcf(), lambda i: patch.set_data(frames[i]), frames=len(frames), interval=50)
    gif.save("results/"+filename+"vid.mp4", fps=60)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default="sac")
    parser.add_argument('--alpha', type=float, default=0.02)
    parser.add_argument('--add_penalty', type=float, default=1)
    parser.add_argument('--keep_add_penalty', type=int, default=0)
    parser.add_argument('--cost_penalty_always', type=int, default=0)
    parser.add_argument('--mult_penalty', type=float, default=-1)
    parser.add_argument('--cost_penalty', type=float, default=-1)
    parser.add_argument('--buckets', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--start_steps', type=int, default=100000)
    parser.add_argument('--split_policy', type=int, default=0)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--safe_policy', type=str, default="")
    parser.add_argument('--name', type=str, default="")

    args = parser.parse_args()
    filename =  datetime.now().strftime("%m_%d_%Y__%H_%M_%S")  + args.name + "/"
    os.mkdir("results/"+filename)
    sys.stdout = open("results/"+filename+"log.txt", 'w')
    print(args)
    if len(args.safe_policy)>0:
        safe_policy=args.safe_policy
    else:
        safe_policy = False
    run_exp(args.alg,args.alpha,args.add_penalty,bool(args.keep_add_penalty),args.mult_penalty,args.cost_penalty,args.buckets,
         args.epochs,args.start_steps,bool(args.cost_penalty_always),bool(args.split_policy),dict(hidden_sizes=[args.hid] * args.l),safe_policy,filename)
