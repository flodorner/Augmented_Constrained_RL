import os
import sys
from test_constraints import run_exp

def experiment_1(): #1 day
    os.mkdir("results/report/experiment_1/")
    name = "sac_a02_ss10k"
    sys.stdout = open("results/report/experiment_1/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=0.2,add_penalty=0,
            epochs=10,start_steps=10000,filename="report/experiment_1/"+ name ,buckets=0)
    name = "sac_a02_ss100k"
    sys.stdout = open("results/report/experiment_1/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=0.2,add_penalty=0,
            epochs=10,start_steps=100000,filename="report/experiment_1/"+ name ,buckets=0)
    name = "sac_a0002_ss10k"
    sys.stdout = open("results/report/experiment_1/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=0.002,add_penalty=0,
            epochs=10,start_steps=10000,filename="report/experiment_1/"+ name ,buckets=0)
    name = "sac_a0002_ss100k"
    sys.stdout = open("results/report/experiment_1/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=0.002,add_penalty=0,
            epochs=10,start_steps=100000,filename="report/experiment_1/"+ name ,buckets=0)
    name = "td3_an01_ss10k"
    sys.stdout = open("results/report/experiment_1/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, add_penalty=0,
            epochs=10, start_steps=10000, filename="report/experiment_1/"+ name , buckets=0, act_noise=0.1)
    name = "td3_an01_ss100k"
    sys.stdout = open("results/report/experiment_1/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, add_penalty=0,
            epochs=10, start_steps=100000, filename="report/experiment_1/"+ name , buckets=0, act_noise=0.1)
    name = "td3_an025_ss10k"
    sys.stdout = open("results/report/experiment_1/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, add_penalty=0,
            epochs=10, start_steps=10000, filename="report/experiment_1/"+ name , buckets=0, act_noise=0.25)
    name = "td3_an025_ss100k"
    sys.stdout = open("results/report/experiment_1/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, add_penalty=0,
            epochs=10, start_steps=100000, filename="report/experiment_1/"+ name , buckets=0, act_noise=0.25)
    return None

def experiment_11():
    os.mkdir("results/report/experiment_11/")
    name = "sac_c_ss10k"
    sys.stdout = open("results/report/experiment_11/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=0,
            epochs=10, start_steps=10000, filename="report/experiment_1/" + name, buckets=0 ,entropy_constraint=-1)
    name = "sac_c_ss100k"
    sys.stdout = open("results/report/experiment_11/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=0,
            epochs=10, start_steps=100000, filename="report/experiment_1/" + name, buckets=0,entropy_constraint=-1)

def experiment_2(): #1-2 days?!
    os.mkdir("results/report/experiment_2/")

    name = "sac_c_ss10k_m0"
    sys.stdout = open("results/report/experiment_2/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=0,mult_penalty=0,
            epochs=10,start_steps=10000,filename="report/experiment_2/"+ name ,entropy_constraint=-1)

    name = "sac_c_ss10k_m001"
    sys.stdout = open("results/report/experiment_2/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=0,mult_penalty=0.01,
            epochs=10,start_steps=10000,filename="report/experiment_2/"+ name ,entropy_constraint=-1)

    name = "td3_ss10k_m0"
    sys.stdout = open("results/report/experiment_2/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=0, mult_penalty=0,
            epochs=10, start_steps=10000, filename="report/experiment_2/" + name)

    name = "td3_ss10k_m001"
    sys.stdout = open("results/report/experiment_2/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=0, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_2/" + name)

    name = "ppo_m0"
    sys.stdout = open("results/report/experiment_2/" + name + "log.txt", 'w')
    run_exp(alg="ppo", steps_per_epoch=25001, add_penalty=0, mult_penalty=0,
            epochs=10, filename="report/experiment_2/" + name)

    name = "ppo_m001"
    sys.stdout = open("results/report/experiment_2/" + name + "log.txt", 'w')
    run_exp(alg="ppo", steps_per_epoch=25001, add_penalty=0, mult_penalty=0.01,
            epochs=10, filename="report/experiment_2/" + name)


    name = "sac_c_ss10k_m001_a1"
    sys.stdout = open("results/report/experiment_2/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=1,mult_penalty=0.01,
            epochs=10,start_steps=10000,filename="report/experiment_2/"+ name ,entropy_constraint=-1)

    name = "sac_c_ss10k_m001_a10"
    sys.stdout = open("results/report/experiment_2/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=10,mult_penalty=0.01,
            epochs=10,start_steps=10000,filename="report/experiment_2/"+ name ,entropy_constraint=-1)

    name = "td3_ss10k_m001_a1"
    sys.stdout = open("results/report/experiment_2/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_2/" + name)

    name = "td3_ss10k_m001_a10"
    sys.stdout = open("results/report/experiment_2/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=10, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_2/" + name)

    name = "ppo_m001_a1"
    sys.stdout = open("results/report/experiment_2/" + name + "log.txt", 'w')
    run_exp(alg="ppo", steps_per_epoch=25001, mult_penalty=0.01, add_penalty=1,
            epochs=10, filename="report/experiment_2/" + name)

    name = "ppo_m001_a10"
    sys.stdout = open("results/report/experiment_2/" + name + "log.txt", 'w')
    run_exp(alg="ppo", steps_per_epoch=25001,mult_penalty=0.01, add_penalty=10,
            epochs=10,  filename="report/experiment_2/" + name)
    return None


def experiment_3(): #1-2 days?!
    os.mkdir("results/report/experiment_3/")

    name = "sac_c_ss10k_m001_a1_c001"
    sys.stdout = open("results/report/experiment_3/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_3/" + name, entropy_constraint=-1,cost_penalty=0.01)

    name = "sac_c_ss10k_m001_a1_c01"
    sys.stdout = open("results/report/experiment_3/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_3/" + name, entropy_constraint=-1,cost_penalty=0.1)

    name = "sac_c_ss10k_m001_a1_c1"
    sys.stdout = open("results/report/experiment_3/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_3/" + name, entropy_constraint=-1,cost_penalty=1)

    name = "td3_ss10k_m001_a1_c001"
    sys.stdout = open("results/report/experiment_3/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_3/" + name, cost_penalty=0.01)

    name = "td3_ss10k_m001_a1_c01"
    sys.stdout = open("results/report/experiment_3/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_3/" + name, cost_penalty=0.1)

    name = "td3_ss10k_m001_a1_c1"
    sys.stdout = open("results/report/experiment_3/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_3/" + name, cost_penalty=1)

    name = "ppo_m001_a1_c001"
    sys.stdout = open("results/report/experiment_3/" + name + "log.txt", 'w')
    run_exp(alg="ppo", steps_per_epoch=25001, mult_penalty=0.01, add_penalty=1,
            epochs=10, filename="report/experiment_3/" + name, cost_penalty=0.01)

    name = "ppo_m001_a1_c01"
    sys.stdout = open("results/report/experiment_3/" + name + "log.txt", 'w')
    run_exp(alg="ppo", steps_per_epoch=25001, mult_penalty=0.01, add_penalty=1,
            epochs=10, filename="report/experiment_3/" + name, cost_penalty=0.1)

    name = "ppo_m001_a1_c1"
    sys.stdout = open("results/report/experiment_3/" + name + "log.txt", 'w')
    run_exp(alg="ppo", steps_per_epoch=25001, mult_penalty=0.01, add_penalty=1,
            epochs=10, filename="report/experiment_3/" + name, cost_penalty=1)

    name = "sac_c_ss10k_m001_a1_c001_nocostinfo"
    sys.stdout = open("results/report/experiment_3/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_3/" + name, entropy_constraint=-1,cost_penalty=0.01,buckets=0)

    name = "td3_ss10k_m001_a1_c001_nocostinfo"
    sys.stdout = open("results/report/experiment_3/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_3/" + name, cost_penalty=0.01,buckets=0)

    name = "ppo_m001_a1_c001_nocostinfo"
    sys.stdout = open("results/report/experiment_3/" + name + "log.txt", 'w')
    run_exp(alg="ppo", steps_per_epoch=25001, mult_penalty=0.01, add_penalty=1,
            epochs=10, filename="report/experiment_3/" + name, cost_penalty=0.01,buckets=0)

    return None

def experiment_4(): #1-2 days?!
    os.mkdir("results/report/experiment_4/")

    name = "ppo_m001_a1_c001_buckets_splitnet"
    sys.stdout = open("results/report/experiment_4/" + name + "log.txt", 'w')
    run_exp(alg="ppo", steps_per_epoch=25001, mult_penalty=0.01, add_penalty=1,
            epochs=10, filename="report/experiment_4/" + name, cost_penalty=0.01,buckets=26,split_policy=True)

    name = "sac_c_ss10k_m001_a1_c001_buckets"
    sys.stdout = open("results/report/experiment_4/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_4/" + name, entropy_constraint=-1,cost_penalty=0.01,buckets=26)

    name = "td3_ss10k_m001_a1_c001_buckets"
    sys.stdout = open("results/report/experiment_4/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_4/" + name, cost_penalty=0.01,buckets=26)

    name = "ppo_m001_a1_c001_buckets"
    sys.stdout = open("results/report/experiment_4/" + name + "log.txt", 'w')
    run_exp(alg="ppo", steps_per_epoch=25001, mult_penalty=0.01, add_penalty=1,
            epochs=10,  filename="report/experiment_4/" + name, cost_penalty=0.01,buckets=26)


    name = "sac_c_ss10k_m001_a1_c001_buckets_splitnet"
    sys.stdout = open("results/report/experiment_4/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_4/" + name, entropy_constraint=-1,cost_penalty=0.01,buckets=26,split_policy=True)

    name = "td3_ss10k_m001_a1_c001_buckets_splitnet"
    sys.stdout = open("results/report/experiment_4/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_4/" + name, cost_penalty=0.01,buckets=26,split_policy=True)

    name = "sac_c_ss10k_m001_a1_c001_buckets_first_layer_512"
    sys.stdout = open("results/report/experiment_4/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_4/" + name, entropy_constraint=-1,cost_penalty=0.01,buckets=26,ac_kwargs={"hidden_sizes":(512,256)})

    name = "td3_ss10k_m001_a1_c001_buckets_first_layer_512"
    sys.stdout = open("results/report/experiment_4/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_4/" + name, cost_penalty=0.01,buckets=26,ac_kwargs={"hidden_sizes":(512,256)})

    name = "ppo_m001_a1_c001_buckets_first_layer_512"
    sys.stdout = open("results/report/experiment_4/" + name + "log.txt", 'w')
    run_exp(alg="ppo", steps_per_epoch=25001, mult_penalty=0.01, add_penalty=1,
            epochs=10, filename="report/experiment_4/" + name, cost_penalty=0.01,buckets=26,ac_kwargs={"hidden_sizes":(512,256)})



def experiment_5(): #2-3 days?!
    #test data augmentation on best identified strategy (sac and td3 if both have a working config).
    #Maybe also test on one other promising parameter configuration.
    return None


# We can do these all in parallel!
def experiment_6(): #1-2 days
    #validate key insights on car goal 1.
    return None

def experiment_7(): #1-2 days
    # validate key insights on point push 1
    return None

def experiment_8():  #1-2 days
    # validate key insights on point goal 2
    return None

def experiment_9(): # 5 days
    # full training on pointgoal1
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=1)
    args = parser.parse_args()
    if args.id == 1:
        experiment_1()
    if args.id == 11:
        experiment_11()
    if args.id == 2:
        experiment_2()
    if args.id == 3:
        experiment_3()
    if args.id == 4:
        experiment_4()
