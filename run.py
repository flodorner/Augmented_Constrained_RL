import os
import sys
from test_constraints import run_exp

#Plan: Run 1,2. Then, 3 and 4 in parallel. 5-8 can be run in parallel as well.
# Make sure you do it early enough to have a buffer in case they take longer than expected or there is some problem.
# Start 4th of jan at latest for 5-8.
# Correspondingly, start 1st of january the lastest for 3 and 4.
# Correspondingly, 3 needs to be started at the 29th (at latest!)

def experiment_1(): #1 day
    os.mkdir("results/report/experiment_1/")
    name = "sac_a02_ss10k"
    sys.stdout = open("results/report/experiment_1/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=0.2,add_penalty=0,
            epochs=10,start_steps=10000,filename="results/report/experiment_1/"+ name ,buckets=0)
    name = "sac_a02_ss100k"
    sys.stdout = open("results/report/experiment_1/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=0.2,add_penalty=0,
            epochs=10,start_steps=100000,filename="results/report/experiment_1/"+ name ,buckets=0)
    name = "sac_a0002_ss10k"
    sys.stdout = open("results/report/experiment_1/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=0.002,add_penalty=0,
            epochs=10,start_steps=10000,filename="results/report/experiment_1/"+ name ,buckets=0)
    name = "sac_a0002_ss100k"
    sys.stdout = open("results/report/experiment_1/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=0.002,add_penalty=0,
            epochs=10,start_steps=100000,filename="results/report/experiment_1/"+ name ,buckets=0)
    name = "td3_an01_ss10k"
    sys.stdout = open("results/report/experiment_1/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, add_penalty=0,
            epochs=10, start_steps=10000, filename="results/report/experiment_1/"+ name , buckets=0, act_noise=0.1)
    name = "td3_an01_ss100k"
    sys.stdout = open("results/report/experiment_1/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, add_penalty=0,
            epochs=10, start_steps=100000, filename="results/report/experiment_1/"+ name , buckets=0, act_noise=0.1)
    name = "td3_an025_ss10k"
    sys.stdout = open("results/report/experiment_1/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, add_penalty=0,
            epochs=10, start_steps=10000, filename="results/report/experiment_1/"+ name , buckets=0, act_noise=0.25)
    name = "td3_an025_ss100k"
    sys.stdout = open("results/report/experiment_1/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, add_penalty=0,
            epochs=10, start_steps=100000, filename="results/report/experiment_1/"+ name , buckets=0, act_noise=0.25)
    return None


    # Run sac with 2 values of alpha, 2 values of start_steps, td3 with two values of start_steps, two different values of act_noise

def experiment_2(): #2-3 days?!
    return None
    # Briefly check if you can speed this up significantly!
    # sac, td3 with normal, deeper, wider and splitnet on alpha=0, ??? (Basically, use best working configuration
    # and have a theoretical reason behind it!). Admit that you found it using informal experimentation

def experiment_3(): #2-3 days?!
   # vary all three parameters along three axes (sac and td3 if both have a working config)
   return None

def experiment_4(): #2-3 days?!
    #test data augmentation on best identified strategy (sac and td3 if both have a working config).
    #Maybe also test on one other promising parameter configuration.
    return None

def experiment_5(): #1-2 days
    #validate key insights on car goal 1.
    return None

def experiment_6(): #1-2 days
    # validate key insights on point push 1
    return None

def experiment_7():  #1-2 days
    # validate key insights on point goal 2
    return None

def experiment_8(): # 5 days
    # full training on pointgoal1
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=1)
    args = parser.parse_args()
    if args.id == 1:
        experiment_1()
