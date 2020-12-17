import safety_gym
import gym
from spinup import sac_pytorch #sac_tf1 should work with tensorflow 1, if you prefer to use that instead
import sys
from wrapper import constraint_wrapper
import pickle

def main(alpha=0.02,filename=""):
    print(alpha)
    env = gym.make('Safexp-PointGoal1-v0')
    env = constraint_wrapper(env)
    sac_pytorch(lambda: env,epochs=125  ,alpha=alpha)

    #Ideally, you would separate train and test runs more directly here rather than reylying on the alg to work exactly as described...
    with open("results/"+filename+"rews.pkl", 'wb') as f:
        pickle.dump(env.total_rews, f)
    with open("results/"+filename+"costs.pkl", 'wb') as f:
        pickle.dump(env.total_costs, f)



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.02)
    args = parser.parse_args()
    filename = "sac_constraint"+str(args.alpha).replace(".","_")
    sys.stdout = open("results/"+filename+".txt", 'w')
    main(args.alpha,filename)
