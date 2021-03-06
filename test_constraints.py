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

def run_exp(alg="sac",alpha=None,add_penalty=1,mult_penalty=None,cost_penalty=0,buckets=None,
         epochs=30,start_steps=10000,split_policy=False,ac_kwargs={"hidden_sizes":(256,256)},
            safe_policy=False,entropy_constraint=-1,collector_policy=None,filename="",steps_per_epoch=10001,
            num_test_episodes=10,act_noise=0.1,data_aug=False,env_name='Safexp-PointGoal1-v0',batch_size=100):

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
    # Collector_policy specifies
    if mult_penalty == -1:
        mult_penalty = None
    if buckets == -1:
        buckets = None
    if entropy_constraint == 0:
        entropy_constraint = None
    if alpha == 0:
        alpha = None

    env = gym.make(env_name) # Create an instance of the safety-gym environment.
    # Create an instance of the constrained environment.
    env = constraint_wrapper(env,add_penalty=add_penalty,mult_penalty=mult_penalty,
                             cost_penalty=cost_penalty,buckets=buckets,safe_policy=safe_policy)
    logger_kwargs = setup_logger_kwargs(filename+"policy",data_dir="results/")
    assert alg == "sac" or alg == "td3" or alg == "ppo"
    # Select learning method
    if alg == "sac":
        import spinup.algos.pytorch.sac.core as core
        if split_policy:
            actor_critic = core.MLPActorCriticSplit
        else:
            actor_critic = core.MLPActorCritic
        # Start training with SAC
        sac_pytorch(lambda: env,epochs=epochs,alpha=alpha,steps_per_epoch=steps_per_epoch,start_steps=start_steps,
                    logger_kwargs=logger_kwargs,num_test_episodes=num_test_episodes,actor_critic=actor_critic,ac_kwargs=ac_kwargs,entropy_constraint=entropy_constraint,collector_policy=collector_policy,data_aug=data_aug,batch_size=batch_size)
    elif alg == "td3":
        import spinup.algos.pytorch.td3.core as core
        if split_policy:
            actor_critic = core.MLPActorCriticSplit
        else:
            actor_critic = core.MLPActorCritic
        # Start training with TD3
        td3_pytorch(lambda: env,epochs=epochs,steps_per_epoch=steps_per_epoch,start_steps=start_steps,logger_kwargs=logger_kwargs,
                    actor_critic=actor_critic,act_noise=act_noise,ac_kwargs=ac_kwargs,collector_policy=collector_policy,data_aug=data_aug,num_test_episodes=num_test_episodes,batch_size=batch_size)
    elif alg == "ppo":
        import spinup.algos.pytorch.ppo.core as core
        assert collector_policy==None
        if split_policy:
            actor_critic = core.MLPActorCriticSplit
        else:
            actor_critic = core.MLPActorCritic
        # Start training with PPO
        ppo_pytorch(lambda: env, epochs=epochs, steps_per_epoch=steps_per_epoch,
                    logger_kwargs=logger_kwargs,
                    actor_critic=actor_critic, ac_kwargs=ac_kwargs)

    #Ideally, you would separate train and test runs more directly here rather than reylying on the alg to work exactly as described...
    # Store training results in pickle file
    with open("results/"+filename+"rews.pkl", 'wb') as f:
        pickle.dump(env.total_rews, f)
    with open("results/"+filename+"costs.pkl", 'wb') as f:
        pickle.dump(env.total_costs, f)



if __name__ == "__main__":
    import argparse
    # Import experiment arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--alg', type=str, default="sac")
    parser.add_argument('--alpha', type=float, default=0)
    parser.add_argument('--add_penalty', type=float, default=1)
    parser.add_argument('--mult_penalty', type=float, default=-1)
    parser.add_argument('--cost_penalty', type=float, default=0)
    parser.add_argument('--buckets', type=int, default=-1)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--start_steps', type=int, default=100000)
    parser.add_argument('--split_policy', type=int, default=0)
    parser.add_argument('--hid', type=int, default=256)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--safe_policy', type=str, default="")
    parser.add_argument('--collector_policy', type=str, default="")
    parser.add_argument('--entropy_constraint', type=int, default=-1)
    parser.add_argument('--env_name', type=str, default='Safexp-PointGoal1-v0')
    parser.add_argument('--name', type=str, default="")

    args = parser.parse_args()
    filename =  datetime.now().strftime("%m_%d_%Y__%H_%M_%S")  + args.name + "/"
    if not os.path.exists("results"): os.mkdir("results")
    if not os.path.exists("results/"+filename): os.mkdir("results/"+filename)
    # Save logs to a file
    sys.stdout = open("results/"+filename+"log.txt", 'w')
    print(args)

    if len(args.safe_policy)>0:
        safe_policy=args.safe_policy
    else:
        safe_policy = False
    if len(args.collector_policy)>0:
        collector_policy=args.collector_policy
    else:
        collector_policy = None
    run_exp(alg=args.alg,alpha=args.alpha,add_penalty=args.add_penalty,
            mult_penalty=args.mult_penalty,cost_penalty=args.cost_penalty,buckets=args.buckets,
         epochs=args.epochs,start_steps=args.start_steps,split_policy=bool(args.split_policy),
            ac_kwargs=dict(hidden_sizes=[args.hid] * args.l),safe_policy=safe_policy,
            entropy_constraint=args.entropy_constraint,collector_policy=collector_policy,filename=filename,data_aug=False,
            env_name=args.env_name)
    # Test the trained policy
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Create an animation of the policy
    _, get_action = load_policy_and_env("results/" + filename + "policy", deterministic=True) # Load policy parameters
    frames = []

    # Create an animation of the policy
    env = gym.make('Safexp-PointGoal1-v0')
    env = constraint_wrapper(env, add_penalty=args.add_penalty, mult_penalty=args.mult_penalty,
                             cost_penalty=args.cost_penalty, buckets=args.buckets,
                             safe_policy=safe_policy)
    # Test for 5k steps
    for i in range(5):
        o = env.reset()
        for i in range(1000):
            frames.append(env.render(mode="rgb_array"))
            a = get_action(torch.tensor(o).to(device))
            o, r, d, _ = env.step(a)
    if args.alg == "sac":
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
    gif.save("results/" + filename + "vid.mp4", fps=60)
