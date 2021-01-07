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
    name = "sac_c_ss10k"
    sys.stdout = open("results/report/experiment_1/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=0,
            epochs=10, start_steps=10000, filename="report/experiment_1/" + name, buckets=0 ,entropy_constraint=-1)
    name = "sac_c_ss100k"
    sys.stdout = open("results/report/experiment_1/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=0,
            epochs=10, start_steps=100000, filename="report/experiment_1/" + name, buckets=0,entropy_constraint=-1)

    return None

def experiment_101():
    os.mkdir("results/report/experiment_101/")
    name = "sac_c_ss10k_pointpush"
    sys.stdout = open("results/report/experiment_101/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=0,
            epochs=10, start_steps=10000, filename="report/experiment_101/" + name, buckets=0, entropy_constraint=-1,
            env_name="Safexp-PointPush1-v0")

    name = "sac_c_ss10k_cargoal"
    sys.stdout = open("results/report/experiment_101/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=0,
            epochs=10, start_steps=10000, filename="report/experiment_101/" + name, buckets=0, entropy_constraint=-1,
            env_name="Safexp-CarGoal1-v0")

    name = "sac_c_ss10k_pointgoal2"
    sys.stdout = open("results/report/experiment_101/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=0,
            epochs=10, start_steps=10000, filename="report/experiment_101/" + name, buckets=0, entropy_constraint=-1,
            env_name="Safexp-PointGoal2-v0")

    return None

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
    os.mkdir("results/report/experiment_5/")

    name = "sac_c_ss10k_m001_a1_c001_data_aug"
    sys.stdout = open("results/report/experiment_5/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_5/" + name, entropy_constraint=-1,cost_penalty=0.01,data_aug=True,buckets=26)

    name = "td3_ss10k_m001_a1_c001_data_aug"
    sys.stdout = open("results/report/experiment_5/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_5/" + name, cost_penalty=0.01,data_aug=True,buckets=26)

    name = "sac_c_ss10k_m001_a1_c01_data_aug"
    sys.stdout = open("results/report/experiment_5/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_5/" + name, entropy_constraint=-1,cost_penalty=0.1,data_aug=True,buckets=26)

    name = "sac_c_ss10k_m001_a1_c1_data_aug"
    sys.stdout = open("results/report/experiment_5/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_5/" + name, entropy_constraint=-1,cost_penalty=1,data_aug=True,buckets=26)

    name = "sac_c_ss10k_m001_a10_data_aug"
    sys.stdout = open("results/report/experiment_5/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=10, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_5/" + name, entropy_constraint=-1,data_aug=True,buckets=26)

    name = "sac_c_ss10k_m001_a10_c001_data_aug"
    sys.stdout = open("results/report/experiment_5/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=10, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_5/" + name, entropy_constraint=-1,data_aug=True,buckets=26,cost_penalty=0.01)

    name = "sac_c_ss10k_m001_a0_c01_data_aug"
    sys.stdout = open("results/report/experiment_5/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=0, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_5/" + name, entropy_constraint=-1,data_aug=True,buckets=26,cost_penalty=0.1)

    name = "sac_c_ss10k_m001_a1_c001_data_aug_nobuckets"
    sys.stdout = open("results/report/experiment_5/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_5/" + name, entropy_constraint=-1,cost_penalty=0.01,data_aug=True)
    return None

def experiment_51():
    os.mkdir("results/report/experiment_51/")

    name = "sac_c_ss10k_m001_a1_c001_long_data_aug"
    sys.stdout = open("results/report/experiment_51/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_51/" + name, entropy_constraint=-1,
            cost_penalty=0.01,data_aug=True,buckets=26)

    name = "sac_c_ss10k_m001_a1_c01_long_data_aug"
    sys.stdout = open("results/report/experiment_51/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_51/" + name, entropy_constraint=-1,
            cost_penalty=0.1,data_aug=True,buckets=26)

    name = "td3_ss10k_m001_a1_c001_long_data_aug"
    sys.stdout = open("results/report/experiment_51/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_51/" + name,
            cost_penalty=0.01,data_aug=True,buckets=26)

    name = "td3_ss10k_m001_a1_c01_long_data_aug"
    sys.stdout = open("results/report/experiment_51/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_51/" + name,
            cost_penalty=0.1,data_aug=True,buckets=26)

    return None

def experiment_52():
    os.mkdir("results/report/experiment_52/")

    name = "sac_c_ss10k_m001_a1_c001_long_data_aug"
    sys.stdout = open("results/report/experiment_52/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_52/" + name, entropy_constraint=-1,
            cost_penalty=0.01, data_aug=True,buckets=26)

    name = "sac_c_ss10k_m001_a1_c001_long_data_aug_bigbatch" #Names are reversed for this and the next experiment.
    # We kept it like this and flagged it in our plotting notebook to avoid additional confusion due to renaming.
    sys.stdout = open("results/report/experiment_52/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_52/" + name, entropy_constraint=-1,
            cost_penalty=0.01,batch_size=500,buckets=26)

    name = "sac_c_ss10k_m001_a1_c001_long_data_bigbatch"
    sys.stdout = open("results/report/experiment_52/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_52/" + name, entropy_constraint=-1,
            cost_penalty=0.01, data_aug=True,batch_size=500,buckets=26)

    name = "sac_c_ss10k_m001_a1_c001_long_bigbatch_correct"
    sys.stdout = open("results/report/experiment_52/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_52/" + name, entropy_constraint=-1,
            cost_penalty=0.01,batch_size=500,buckets=26)

    name = "td3_ss10k_m001_a1_c001_long_bigbatch"
    sys.stdout = open("results/report/experiment_52/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_52/" + name,
            cost_penalty=0.01,batch_size=500,buckets=26)

    name = "td3_ss10k_m001_a1_c001_long_data_aug_bigbatch"
    sys.stdout = open("results/report/experiment_52/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_52/" + name,
            cost_penalty=0.01, data_aug=True,batch_size=500,buckets=26)

    name = "td3_ss10k_m001_a1_c001_data_aug_nobuckets"
    sys.stdout = open("results/report/experiment_52/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_52/" + name ,cost_penalty=0.01,data_aug=True)

    return None


def experiment_57():
    os.mkdir("results/report/experiment_57/")

    name = "sac_c_ss10k_m001_a1_c001_long_data_aug_cargoal"
    sys.stdout = open("results/report/experiment_57/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_57/" + name, entropy_constraint=-1,
            cost_penalty=0.01,data_aug=True,buckets=26,env_name="Safexp-CarGoal1-v0")

    name = "sac_c_ss10k_m001_a1_c01_long_data_aug_cargoal"
    sys.stdout = open("results/report/experiment_57/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_57/" + name, entropy_constraint=-1,
            cost_penalty=0.1,data_aug=True,buckets=26,env_name="Safexp-CarGoal1-v0")

    name = "td3_ss10k_m001_a1_c001_long_data_aug_cargoal"
    sys.stdout = open("results/report/experiment_57/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_57/" + name,
            cost_penalty=0.01,data_aug=True,buckets=26,env_name="Safexp-CarGoal1-v0")

    name = "td3_ss10k_m001_a1_c01_long_data_aug_cargoal"
    sys.stdout = open("results/report/experiment_57/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_57/" + name,
            cost_penalty=0.1,data_aug=True,buckets=26,env_name="Safexp-CarGoal1-v0")

    name = "sac_c_ss10k_m001_a1_c001_data_aug_cont_cargoal"
    sys.stdout = open("results/report/experiment_57/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_57/" + name, entropy_constraint=-1,
            cost_penalty=0.01,data_aug=True,env_name="Safexp-CarGoal1-v0")

    name = "td3_ss10k_m001_a1_c001_data_aug_cont_cargoal"
    sys.stdout = open("results/report/experiment_57/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_57/" + name,
            cost_penalty=0.01,data_aug=True,env_name="Safexp-CarGoal1-v0")
    return None


def experiment_58():
    os.mkdir("results/report/experiment_58/")

    name = "sac_c_ss10k_m001_a1_c001_long_data_aug_pointpush"
    sys.stdout = open("results/report/experiment_58/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_58/" + name, entropy_constraint=-1,
            cost_penalty=0.01,data_aug=True,buckets=26,env_name="Safexp-PointPush1-v0")

    name = "sac_c_ss10k_m001_a1_c01_long_data_aug_pointpush"
    sys.stdout = open("results/report/experiment_58/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_58/" + name, entropy_constraint=-1,
            cost_penalty=0.1,data_aug=True,buckets=26,env_name="Safexp-PointPush1-v0")

    name = "td3_ss10k_m001_a1_c001_long_data_aug_pointpush"
    sys.stdout = open("results/report/experiment_58/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_58/" + name,
            cost_penalty=0.01,data_aug=True,buckets=26,env_name="Safexp-PointPush1-v0")

    name = "td3_ss10k_m001_a1_c01_long_data_aug_pointpush"
    sys.stdout = open("results/report/experiment_58/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_58/" + name,
            cost_penalty=0.1,data_aug=True,buckets=26,env_name="Safexp-PointPush1-v0")

    name = "sac_c_ss10k_m001_a1_c001_data_aug_cont_pointpush"
    sys.stdout = open("results/report/experiment_58/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_58/" + name, entropy_constraint=-1,
            cost_penalty=0.01,data_aug=True,env_name="Safexp-PointPush1-v0")

    name = "td3_ss10k_m001_a1_c001_data_aug_cont_pointpush"
    sys.stdout = open("results/report/experiment_58/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_58/" + name,
            cost_penalty=0.01,data_aug=True,env_name="Safexp-PointPush1-v0")

    return None


def experiment_59():
    os.mkdir("results/report/experiment_59/")

    name = "sac_c_ss10k_m001_a1_c001_long_data_aug_pointgoal2"
    sys.stdout = open("results/report/experiment_59/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_59/" + name, entropy_constraint=-1,
            cost_penalty=0.01,data_aug=True,buckets=26,env_name="Safexp-PointGoal2-v0")

    name = "sac_c_ss10k_m001_a1_c01_long_data_aug_pointgoal2"
    sys.stdout = open("results/report/experiment_59/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_59/" + name, entropy_constraint=-1,
            cost_penalty=0.1,data_aug=True,buckets=26,env_name="Safexp-PointGoal2-v0")

    name = "td3_ss10k_m001_a1_c001_long_data_aug_pointgoal2"
    sys.stdout = open("results/report/experiment_59/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_59/" + name,
            cost_penalty=0.01,data_aug=True,buckets=26,env_name="Safexp-PointGoal2-v0")

    name = "td3_ss10k_m001_a1_c01_long_data_aug_pointgoal2"
    sys.stdout = open("results/report/experiment_59/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_59/" + name,
            cost_penalty=0.1,data_aug=True,buckets=26,env_name="Safexp-PointGoal2-v0")

    name = "sac_c_ss10k_m001_a1_c001_data_aug_cont_pointgoal2"
    sys.stdout = open("results/report/experiment_59/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_59/" + name, entropy_constraint=-1,
            cost_penalty=0.01,data_aug=True,env_name="Safexp-PointGoal2-v0")

    name = "td3_ss10k_m001_a1_c001_data_aug_cont_pointgoal2"
    sys.stdout = open("results/report/experiment_59/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_59/" + name,
            cost_penalty=0.01,data_aug=True,env_name="Safexp-PointGoal2-v0")

    return None





def experiment_6():
    os.mkdir("results/report/experiment_6/")

    name = "sac_c_ss10k_m1_a2"
    sys.stdout = open("results/report/experiment_6/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=2,mult_penalty=1,
            epochs=10,start_steps=10000,filename="report/experiment_6/"+ name ,entropy_constraint=-1)
    name = "sac_c_ss10k_m1_a5"
    sys.stdout = open("results/report/experiment_6/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=5,mult_penalty=1,
            epochs=10,start_steps=10000,filename="report/experiment_6/"+ name ,entropy_constraint=-1)
    name = "sac_c_ss10k_m1_a10"
    sys.stdout = open("results/report/experiment_6/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=10,mult_penalty=1,
            epochs=10,start_steps=10000,filename="report/experiment_6/"+ name ,entropy_constraint=-1)

    name = "sac_c_ss10k_m001_a1_long"
    sys.stdout = open("results/report/experiment_6/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=1,mult_penalty=0.01,
            epochs=20,start_steps=10000,filename="report/experiment_6/"+ name ,entropy_constraint=-1)

    name = "sac_c_ss10k_m001_a10_long"
    sys.stdout = open("results/report/experiment_6/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=10,mult_penalty=0.01,
            epochs=20,start_steps=10000,filename="report/experiment_6/"+ name ,entropy_constraint=-1)

    name = "sac_c_ss10k_m001_a1_c001_long"
    sys.stdout = open("results/report/experiment_6/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_6/" + name, entropy_constraint=-1,
            cost_penalty=0.01)

    name = "sac_c_ss10k_m001_a1_c01_long"
    sys.stdout = open("results/report/experiment_6/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_6/" + name, entropy_constraint=-1,
            cost_penalty=0.1)

    name = "sac_c_ss10k_m001_a0_c01_long"
    sys.stdout = open("results/report/experiment_6/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=0, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_6/" + name, entropy_constraint=-1,
            cost_penalty=0.1)
    
    return None

def experiment_7(): #1-2 days
    os.mkdir("results/report/experiment_7/")

    name = "sac_c_ss10k_m001_a1_c001_cargoal"
    sys.stdout = open("results/report/experiment_7/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_7/" + name, entropy_constraint=-1,cost_penalty=0.01,env_name="Safexp-CarGoal1-v0")

    name = "td3_ss10k_m001_a1_c001_cargoal"
    sys.stdout = open("results/report/experiment_7/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_7/" + name, cost_penalty=0.01,env_name="Safexp-CarGoal1-v0")

    name = "ppo_m001_a1_c001_cargoal"
    sys.stdout = open("results/report/experiment_7/" + name + "log.txt", 'w')
    run_exp(alg="ppo", steps_per_epoch=25001, mult_penalty=0.01, add_penalty=1,
            epochs=10, filename="report/experiment_7/" + name, cost_penalty=0.01,env_name="Safexp-CarGoal1-v0")

    name = "sac_c_ss10k_m001_a1_c001_buckets_cargoal"
    sys.stdout = open("results/report/experiment_7/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_7/" + name, entropy_constraint=-1,cost_penalty=0.01,buckets=26,env_name="Safexp-CarGoal1-v0")

    name = "sac_c_ss10k_m001_a1_c001_buckets_first_layer_512_cargoal"
    sys.stdout = open("results/report/experiment_7/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_7/" + name, entropy_constraint=-1,cost_penalty=0.01,buckets=26,ac_kwargs={"hidden_sizes":(512,256)},env_name="Safexp-CarGoal1-v0")

    name = "sac_c_ss10k_m1_a2_cargoal"
    sys.stdout = open("results/report/experiment_7/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=2,mult_penalty=1,
            epochs=10,start_steps=10000,filename="report/experiment_7/"+ name ,entropy_constraint=-1,env_name="Safexp-CarGoal1-v0")
    name = "sac_c_ss10k_m1_a5_cargoal"
    sys.stdout = open("results/report/experiment_7/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=5,mult_penalty=1,
            epochs=10,start_steps=10000,filename="report/experiment_7/"+ name ,entropy_constraint=-1,env_name="Safexp-CarGoal1-v0")
    name = "sac_c_ss10k_m1_a10_cargoal"
    sys.stdout = open("results/report/experiment_7/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=10,mult_penalty=1,
            epochs=10,start_steps=10000,filename="report/experiment_7/"+ name ,entropy_constraint=-1,env_name="Safexp-CarGoal1-v0")

    name = "sac_c_ss10k_m001_a1_cargoal"
    sys.stdout = open("results/report/experiment_7/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=1,mult_penalty=0.01,
            epochs=20,start_steps=10000,filename="report/experiment_7/"+ name ,entropy_constraint=-1,env_name="Safexp-CarGoal1-v0")

    name = "sac_c_ss10k_m001_a10_cargoal"
    sys.stdout = open("results/report/experiment_7/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=10,mult_penalty=0.01,
            epochs=20,start_steps=10000,filename="report/experiment_7/"+ name ,entropy_constraint=-1,env_name="Safexp-CarGoal1-v0")

    name = "sac_c_ss10k_m001_a1_c001_cargoal"
    sys.stdout = open("results/report/experiment_7/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_7/" + name, entropy_constraint=-1,
            cost_penalty=0.01,env_name="Safexp-CarGoal1-v0")

    name = "sac_c_ss10k_m001_a1_c01_cargoal"
    sys.stdout = open("results/report/experiment_7/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_7/" + name, entropy_constraint=-1,
            cost_penalty=0.1,env_name="Safexp-CarGoal1-v0")

    name = "sac_c_ss10k_m001_a0_c01_cargoal"
    sys.stdout = open("results/report/experiment_7/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=0, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_7/" + name, entropy_constraint=-1,
            cost_penalty=0.1,env_name="Safexp-CarGoal1-v0")
    
    #validate key insights on car goal 1.
    return None

def experiment_8(): #1-2 days
    os.mkdir("results/report/experiment_8/")

    name = "sac_c_ss10k_m001_a1_c001_pointpush"
    sys.stdout = open("results/report/experiment_8/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_8/" + name, entropy_constraint=-1,cost_penalty=0.01,env_name="Safexp-PointPush1-v0")

    name = "td3_ss10k_m001_a1_c001_pointpush"
    sys.stdout = open("results/report/experiment_8/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_8/" + name, cost_penalty=0.01,env_name="Safexp-PointPush1-v0")

    name = "ppo_m001_a1_c001_pointpush"
    sys.stdout = open("results/report/experiment_8/" + name + "log.txt", 'w')
    run_exp(alg="ppo", steps_per_epoch=25001, mult_penalty=0.01, add_penalty=1,
            epochs=10, filename="report/experiment_8/" + name, cost_penalty=0.01,env_name="Safexp-PointPush1-v0")

    name = "sac_c_ss10k_m001_a1_c001_buckets_pointpush"
    sys.stdout = open("results/report/experiment_8/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_8/" + name, entropy_constraint=-1,cost_penalty=0.01,buckets=26,env_name="Safexp-PointPush1-v0")

    name = "sac_c_ss10k_m001_a1_c001_buckets_first_layer_512_pointpush"
    sys.stdout = open("results/report/experiment_8/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_8/" + name, entropy_constraint=-1,cost_penalty=0.01,buckets=26,ac_kwargs={"hidden_sizes":(512,256)},env_name="Safexp-PointPush1-v0")

    name = "sac_c_ss10k_m1_a2_pointpush"
    sys.stdout = open("results/report/experiment_8/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=2,mult_penalty=1,
            epochs=10,start_steps=10000,filename="report/experiment_8/"+ name ,entropy_constraint=-1,env_name="Safexp-PointPush1-v0")
    name = "sac_c_ss10k_m1_a5_pointpush"
    sys.stdout = open("results/report/experiment_8/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=5,mult_penalty=1,
            epochs=10,start_steps=10000,filename="report/experiment_8/"+ name ,entropy_constraint=-1,env_name="Safexp-PointPush1-v0")
    name = "sac_c_ss10k_m1_a10_pointpush"
    sys.stdout = open("results/report/experiment_8/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=10,mult_penalty=1,
            epochs=10,start_steps=10000,filename="report/experiment_8/"+ name ,entropy_constraint=-1,env_name="Safexp-PointPush1-v0")

    name = "sac_c_ss10k_m001_a1_pointpush"
    sys.stdout = open("results/report/experiment_8/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=1,mult_penalty=0.01,
            epochs=20,start_steps=10000,filename="report/experiment_8/"+ name ,entropy_constraint=-1,env_name="Safexp-PointPush1-v0")

    name = "sac_c_ss10k_m001_a10_pointpush"
    sys.stdout = open("results/report/experiment_8/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=10,mult_penalty=0.01,
            epochs=20,start_steps=10000,filename="report/experiment_8/"+ name ,entropy_constraint=-1,env_name="Safexp-PointPush1-v0")

    name = "sac_c_ss10k_m001_a1_c001_pointpush"
    sys.stdout = open("results/report/experiment_8/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_8/" + name, entropy_constraint=-1,
            cost_penalty=0.01,env_name="Safexp-PointPush1-v0")

    name = "sac_c_ss10k_m001_a1_c01_pointpush"
    sys.stdout = open("results/report/experiment_8/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_8/" + name, entropy_constraint=-1,
            cost_penalty=0.1,env_name="Safexp-PointPush1-v0")

    name = "sac_c_ss10k_m001_a0_c01_pointpush"
    sys.stdout = open("results/report/experiment_8/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=0, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_8/" + name, entropy_constraint=-1,
            cost_penalty=0.1,env_name="Safexp-PointPush1-v0")
    # validate key insights on point push 1
    return None

def experiment_9():  #1-2 days
    os.mkdir("results/report/experiment_9/")
    
    name = "sac_c_ss10k_m001_a1_c001_pointgoal2"
    sys.stdout = open("results/report/experiment_9/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_9/" + name, entropy_constraint=-1,cost_penalty=0.01,env_name="Safexp-PointGoal2-v0")

    name = "td3_ss10k_m001_a1_c001_pointgoal2"
    sys.stdout = open("results/report/experiment_9/" + name + "log.txt", 'w')
    run_exp(alg="td3", steps_per_epoch=25001, num_test_episodes=25, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_9/" + name, cost_penalty=0.01,env_name="Safexp-PointGoal2-v0")

    name = "ppo_m001_a1_c001_pointgoal2"
    sys.stdout = open("results/report/experiment_9/" + name + "log.txt", 'w')
    run_exp(alg="ppo", steps_per_epoch=25001, mult_penalty=0.01, add_penalty=1,
            epochs=10, filename="report/experiment_9/" + name, cost_penalty=0.01,env_name="Safexp-PointGoal2-v0")

    name = "sac_c_ss10k_m001_a1_c001_buckets_pointgoal2"
    sys.stdout = open("results/report/experiment_9/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_9/" + name, entropy_constraint=-1,cost_penalty=0.01,buckets=26,env_name="Safexp-PointGoal2-v0")

    name = "sac_c_ss10k_m001_a1_c001_buckets_first_layer_512_pointgoal2"
    sys.stdout = open("results/report/experiment_9/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=10, start_steps=10000, filename="report/experiment_9/" + name, entropy_constraint=-1,cost_penalty=0.01,buckets=26,ac_kwargs={"hidden_sizes":(512,256)},env_name="Safexp-PointGoal2-v0")

    name = "sac_c_ss10k_m1_a2_pointgoal2"
    sys.stdout = open("results/report/experiment_9/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=2,mult_penalty=1,
            epochs=10,start_steps=10000,filename="report/experiment_9/"+ name ,entropy_constraint=-1,env_name="Safexp-PointGoal2-v0")
    name = "sac_c_ss10k_m1_a5_pointgoal2"
    sys.stdout = open("results/report/experiment_9/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=5,mult_penalty=1,
            epochs=10,start_steps=10000,filename="report/experiment_9/"+ name ,entropy_constraint=-1,env_name="Safexp-PointGoal2-v0")
    name = "sac_c_ss10k_m1_a10_pointgoal2"
    sys.stdout = open("results/report/experiment_9/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=10,mult_penalty=1,
            epochs=10,start_steps=10000,filename="report/experiment_9/"+ name ,entropy_constraint=-1,env_name="Safexp-PointGoal2-v0")

    name = "sac_c_ss10k_m001_a1_pointgoal2"
    sys.stdout = open("results/report/experiment_9/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=1,mult_penalty=0.01,
            epochs=20,start_steps=10000,filename="report/experiment_9/"+ name ,entropy_constraint=-1,env_name="Safexp-PointGoal2-v0")

    name = "sac_c_ss10k_m001_a10_pointgoal2"
    sys.stdout = open("results/report/experiment_9/"+ name + "log.txt", 'w')
    run_exp(alg="sac",steps_per_epoch=25001,num_test_episodes=25,alpha=None,add_penalty=10,mult_penalty=0.01,
            epochs=20,start_steps=10000,filename="report/experiment_9/"+ name ,entropy_constraint=-1,env_name="Safexp-PointGoal2-v0")

    name = "sac_c_ss10k_m001_a1_c001_pointgoal2"
    sys.stdout = open("results/report/experiment_9/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_9/" + name, entropy_constraint=-1,
            cost_penalty=0.01,env_name="Safexp-PointGoal2-v0")

    name = "sac_c_ss10k_m001_a1_c01_pointgoal2"
    sys.stdout = open("results/report/experiment_9/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_9/" + name, entropy_constraint=-1,
            cost_penalty=0.1,env_name="Safexp-PointGoal2-v0")

    name = "sac_c_ss10k_m001_a0_c01_pointgoal2"
    sys.stdout = open("results/report/experiment_9/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=0, mult_penalty=0.01,
            epochs=20, start_steps=10000, filename="report/experiment_9/" + name, entropy_constraint=-1,
            cost_penalty=0.1,env_name="Safexp-PointGoal2-v0")

    # validate key insights on point goal 2
    return None


def experiment_10(): # 5 days





    # Validate data augmentation on other environments!
    return None


def experiment_11(): # 5 days
    os.mkdir("results/report/experiment_11/")
    name = "sac_c_ss10k_m001_a1_c001_10M"
    sys.stdout = open("results/report/experiment_11/" + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=25001, num_test_episodes=25, alpha=None, add_penalty=1, mult_penalty=0.01,
            epochs=400, start_steps=10000, filename="report/experiment_11/" + name, entropy_constraint=-1,
            cost_penalty=0.01)
    return None

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=1)
    args = parser.parse_args()
    if args.id == 1:
        experiment_1()
    if args.id == 101:
        experiment_101()
    if args.id == 2:
        experiment_2()
    if args.id == 3:
        experiment_3()
    if args.id == 4:
        experiment_4()
    if args.id == 5:
        experiment_5()
    if args.id == 51:
        experiment_51()
    if args.id == 52:
        experiment_52()
    if args.id == 57:
        experiment_57()
    if args.id == 58:
        experiment_58()
    if args.id == 59:
        experiment_59()
    if args.id == 6:
        experiment_6()
    if args.id == 7:
        experiment_7()
    if args.id == 8:
        experiment_8()
    if args.id == 9:
        experiment_9()
    if args.id == 10:
        experiment_10()
    if args.id == 11:
        experiment_11()

