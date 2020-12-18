import safety_gym
import gym
from spinup import sac_pytorch #sac_tf1 should work with tensorflow 1, if you prefer to use that instead
import sys
from wrapper import constraint_wrapper
import pickle

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
    sac_pytorch(lambda: env,epochs=30,alpha=alpha,steps_per_epoch=10001)

    #Ideally, you would separate train and test runs more directly here rather than reylying on the alg to work exactly as described...
    with open("results/"+filename+"rews.pkl", 'wb') as f:
        pickle.dump(env.total_rews, f)
    with open("results/"+filename+"costs.pkl", 'wb') as f:
        pickle.dump(env.total_costs, f)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.02)
    parser.add_argument('--add_penalty', type=float, default=1)
    parser.add_argument('--keep_add_penalty', type=bool, default=False)
    parser.add_argument('--mult_penalty', type=float, default=-1)
    parser.add_argument('--cost_penalty', type=float, default=-1)
    parser.add_argument('--buckets', type=float, default=-1)
    args = parser.parse_args()
    filename = "sac_constraint_a_"+str(args.alpha).replace(".","_")+"_pen_"+str(args.add_penalty).replace(".","_")\
               +args.keep_add_penalty*"_s"+"_mult_"+str(args.mult_penalty).replace(".","_")+"_cost_"+\
               str(args.cost_penalty).replace(".","_")+"_buckets_"+str(args.buckets).replace(".","_")
    sys.stdout = open("results/"+filename+".txt", 'w')
    main(args.alpha,args.add_penalty,args.keep_add_penalty,args.mult_penalty,args.cost_penalty,args.buckets,filename)
