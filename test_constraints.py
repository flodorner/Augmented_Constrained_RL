import safety_gym
import gym
from spinup import sac_pytorch #sac_tf1 should work with tensorflow 1, if you prefer to use that instead
import sys
from wrapper import constraint_wrapper
import pickle

def main(alpha=0.02,add_penalty=1,keep_add_penalty=True,mult_penalty=None,filename=""):
    if mult_penalty == -1:
        mult_penalty = None
    env = gym.make('Safexp-PointGoal1-v0')
    env = constraint_wrapper(env,add_penalty=add_penalty,keep_add_penalty=keep_add_penalty,mult_penalty=mult_penalty)
    sac_pytorch(lambda: env,epochs=125,alpha=alpha)

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
    parser.add_argument('--keep_add_penalty', type=bool, default=True)
    parser.add_argument('--mult_penalty', type=float, default=-1)
    args = parser.parse_args()
    filename = "sac_constraint_a_"+str(args.alpha).replace(".","_")+"_pen_"+str(args.add_penalty).replace(".","_")+args.keep_add_penalty*"_s"+"_mult_"+str(args.mult_penalty).replace(".","_")
    sys.stdout = open("results/"+filename+".txt", 'w')
    main(args.alpha,args.add_penalty,args.keep_add_penalty,args.mult_penalty,filename)
