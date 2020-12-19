import safety_gym
import gym
from spinup import sac_pytorch #sac_tf1 should work with tensorflow 1, if you prefer to use that instead
import sys
from wrapper import constraint_wrapper
import pickle
from spinup.utils.run_utils import setup_logger_kwargs
from spinup.utils.test_policy import load_policy_and_env, run_policy
import matplotlib.pyplot as plt
from matplotlib import animation

def main(alpha=0.02,add_penalty=1,keep_add_penalty=True,mult_penalty=None,cost_penalty=None,buckets=None,filename=""):
    if mult_penalty == -1:
        mult_penalty = None
    if cost_penalty == -1:
        cost_penalty = None
    if buckets == -1:
        buckets = None
    env = gym.make('Safexp-PointGoal1-v0')
    env = constraint_wrapper(env,add_penalty=add_penalty,keep_add_penalty=keep_add_penalty,mult_penalty=mult_penalty,
                             cost_penalty=cost_penalty,buckets=buckets)
    logger_kwargs = setup_logger_kwargs(filename,data_dir="results/")
    sac_pytorch(lambda: env,epochs=30,alpha=alpha,steps_per_epoch=10001,logger_kwargs=logger_kwargs)

    #Ideally, you would separate train and test runs more directly here rather than reylying on the alg to work exactly as described...
    with open("results/"+filename+"rews.pkl", 'wb') as f:
        pickle.dump(env.total_rews, f)
    with open("results/"+filename+"costs.pkl", 'wb') as f:
        pickle.dump(env.total_costs, f)

    _, get_action = load_policy_and_env("results/"+filename)
    o = env.reset()
    frames = []

    for i in range(1000):
        frames.append(env.render(mode="rgb_array"))
        a = get_action(o)
        o, r, d, _ = env.step(a)
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi=72)
    plt.axis('off')
    patch = plt.imshow(frames[0])
    gif = animation.FuncAnimation(plt.gcf(), lambda i: patch.set_data(frames[i]), frames=len(frames), interval=50)
    gif.save("results/"+filename+".mp4", fps=60)



    #render and save video of final agent?!


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.02)
    parser.add_argument('--add_penalty', type=float, default=1)
    parser.add_argument('--keep_add_penalty', type=bool, default=False)
    parser.add_argument('--mult_penalty', type=float, default=-1)
    parser.add_argument('--cost_penalty', type=float, default=-1)
    parser.add_argument('--buckets', type=int, default=-1)
    args = parser.parse_args()
    filename = "sac_constraint_a_"+str(args.alpha).replace(".","_")+"_pen_"+str(args.add_penalty).replace(".","_")\
               +args.keep_add_penalty*"_s"+"_mult_"+str(args.mult_penalty).replace(".","_")+"_cost_"+\
               str(args.cost_penalty).replace(".","_")+"_buckets_"+str(args.buckets).replace(".","_")
    sys.stdout = open("results/"+filename+".txt", 'w')
    main(args.alpha,args.add_penalty,args.keep_add_penalty,args.mult_penalty,args.cost_penalty,args.buckets,filename)
