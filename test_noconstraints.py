#Test if SAC is able to deal with the safety-gym environments when constraints are ignored.

import safety_gym
import gym
from spinup import sac_pytorch #sac_tf1 should work with tensorflow 1, if you prefer to use that instead
import sys

# Train an agent with SAC ignoring cost constraints
def main(alpha=0.2):
    print(alpha)
    # Create Safety-Gym environment instance
    env = gym.make('Safexp-PointGoal1-v0')
    env.stepb = env.step
    def res_step(action):
        if env.done:
            env.reset()
        return env.stepb(action)

    env.step = res_step
    #Start training
    sac_pytorch(lambda: env,epochs=250,alpha=alpha)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--alpha', type=float, default=0.2)
    args = parser.parse_args()
    # Save logs to a file
    sys.stdout = open("sac"+str(args.alpha).replace(".","_")+".txt", 'w')
    main(args.alpha)
