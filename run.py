import os
import sys
from test_constraints import run_exp

# Check if directory exists before creating
def checked_mkdir(path):
    if not os.path.exists(path):
        os.mkdir(path)

def experiment_1():
    #Test different hyperparameters for SAC and TD3 on the unconstrained PointGoal1 task
    folder = "results/report/experiment_1/"
    checked_mkdir(folder)

    name = "sac_adaptive"
    sys.stdout = open(folder + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=10001, alpha=None, epochs=100, start_steps=10000, filename=folder + name,
            buckets=26,entropy_constraint=-1,adaptive=[0 for i in range(100)]+[100 for i in range(100)],adaptive_len=10,
            max_penalty=100)

    name = "sac_adaptive_m0"
    sys.stdout = open(folder + name + "log.txt", 'w')
    run_exp(alg="sac", steps_per_epoch=10001, alpha=None, epochs=100, start_steps=10000, filename=folder + name,
            buckets=26,entropy_constraint=-1,adaptive=[0 for i in range(100)]+[100 for i in range(100)],adaptive_len=10,
            max_penalty=100,mult_penalty=0)

    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--id', type=int, default=1)
    args = parser.parse_args()
    if not os.path.exists("results"): checked_mkdir("results")
    if not os.path.exists("results/report"): checked_mkdir("results/report")
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
    if args.id == 61:
        experiment_61()
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
    if args.id == 12:
        experiment_12()
    if args.id == 13:
        experiment_13()
    if args.id == 14:
        experiment_14()
    if args.id == 15:
        experiment_15()
    if args.id == 16:
        experiment_16()
    if args.id == 17:
        experiment_17()
