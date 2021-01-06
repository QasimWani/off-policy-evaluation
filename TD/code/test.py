### Graph OPE outputs according to different policies, π
import ope
import utils
import argparse
import subprocess
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt


# Main function to graph plots according to user params

def run_model(args:dict):
    """
    Plot test & training scores w.r.t:
    1. Importance Sampling algorithm.
    2. Wrapper function algorithm for MSE approximation.
    @Param:
    - args: config list for running ope.py (run python ope.py -h for info)
    """

    #run comparison with RIS and OIS
    if(args.IS == 'true'):
        compare_IS(args)
    elif(args.IS == 'false'):
        compare_mse(args)
    else:
        raise ValueError("Incorrect option provided. Run python test.py -h for info")
        
        
        
def compare_mse(args:dict, mode='RIS'):
    """ Compare different MSE algorithms """
    algorithms = ["lasso", "lr", "maml", "poly", "ridge"]
    mse_alg = algorithms if args.mse == [] else utils.to_list(args.mse, str) #select specfic MSE algorithm(s) if user provided
    
    p_train = []; p_test = [] #root error holder

    for alg in tqdm(mse_alg):
        train_error = []; test_error = [] #wrapper function specific error holder
        for agent_idx in range(1, 10): #number of agents, see specifications in ../model/README.md
            cmd = f'python ope.py -agents={agent_idx},{args.agents} -mse={alg} -IS={mode}'
            result = subprocess.check_output(cmd, shell=True).splitlines()[-3:-1]

            #compute errors
            tre = float(str(result[0]).split(" ")[-1].strip()[:-1])
            tee = float(str(result[1]).split(" ")[-1].strip()[:-1])
            
            train_error.append(tre)
            test_error.append(tee)
        
        #--update entries--
        p_train.append(train_error)
        p_test.append(test_error)
    
    ### Plot results
    for i, alg_score in enumerate(p_test):
        print(f'{mse_alg[i]} function:\nTrain:\n{p_train[i]}\nTest:\n{alg_score}\n\n')
        plt.plot(np.arange(1, 10), alg_score, label=f'{mse_alg[i]}')
    
    plt.title("MSE Error for different behavior agents")
    plt.xlabel("Policy strength")
    plt.ylabel("MSE Error")
    plt.legend()
    plt.grid()
    plt.show()
    
    
def compare_IS(args:dict):
    """ Compare different Importance Sampling ratios """
    mse_alg = 'lr' if args.mse == [] else utils.to_list(args.mse, str)[0] #select first MSE algorithm if multiple provided
            
    train_error_OIS = []; test_error_OIS = []
    train_error_RIS = []; test_error_RIS = []
    
    for agent_idx in tqdm(range(1, 10)): #number of agents, see specifications in ../model/README.md
                    
        cmd = f'python ope.py -agents={agent_idx},{args.agents} -mse={mse_alg} -IS=RIS'
        result_RIS = subprocess.check_output(cmd, shell=True).splitlines()[-3:-1]
        
        cmd = f'python ope.py -agents={agent_idx},{args.agents} -mse={mse_alg} -IS=OIS'
        result_OIS = subprocess.check_output(cmd, shell=True).splitlines()[-3:-1]
        
        #compute RIS errors
        
        tre = float(str(result_RIS[0]).split(" ")[-1].strip()[:-1])
        tee = float(str(result_RIS[1]).split(" ")[-1].strip()[:-1])

        train_error_RIS.append(tre)
        test_error_RIS.append(tee)

        #compute OIS errors
        tre = float(str(result_OIS[0]).split(" ")[-1].strip()[:-1])
        tee = float(str(result_OIS[1]).split(" ")[-1].strip()[:-1])
        
        train_error_OIS.append(tre)
        test_error_OIS.append(tee)
        
    print("RIS Test errors",test_error_RIS)
    print("OIS Test errors", test_error_OIS)
    
    # plt.plot(train_error_RIS, label='RIS Train Errors')
    plt.plot(np.arange(1, 10), test_error_RIS, label='RIS Test Errors')
    
    # plt.plot(train_error_OIS, label='OIS Train Errors')
    plt.plot(np.arange(1, 10), test_error_OIS, label='OIS Test Errors')

    plt.title("MSE Error for different behavior agents")
    plt.xlabel("Policy strength")
    plt.ylabel("MSE Error")
    plt.legend()
    plt.grid()
    plt.show()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OPE test simulation for TD learning')

    #comparison between agents
    parser.add_argument('-agents', metavar='List_agents', type=str, default='10',
                        help='list of agents to store in policy matrix.')
    #flag for IS comparison
    parser.add_argument('-IS', metavar='isCompareIS', type=str, default="true",
                        help='Compare MSE error w.r.t IS function? true[false] = compare[skip]')
    #type of wrapper function for MSE approximation
    parser.add_argument('-mse', metavar='compareMSE', type=str, default=[],
                        help='Enter desired wrapper algorithm for comparison of MSE error. Leave blank to include all. Options = '\
                        'lr: linear regression; poly: polynomial regression; ridge: Ridge Regression; lasso: Lasso Regression; maml: Model Agnostic Meta-Learning')
    
    args = parser.parse_args() #parse arguments
    
    #Simulate!
    run_model(args)