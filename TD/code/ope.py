import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import time
import tqdm

from collections import defaultdict
from os import listdir, getcwd
from os.path import isabs, join, isfile
import re
import argparse

from model import Agent
import utils
from estimator import IS
from wrapper import MSE
from maml import MAML

def load_policies(path):
    """Loads all policies in a directory"""
    cwd = getcwd() #get current working directory
    index = lambda idx : int(re.findall(r'[0-9]+', idx)[0]) #get policy number
    def model(f):
        policy = utils.load_policy(f) if ".npy" in f else None #load agent
        if(policy is None):
            return None
        agx = Agent(env)
        agx.Q = policy
        return agx
    
    return {index(f): model(join(path, f)) for f in listdir(join(cwd, path)) if isabs(join(cwd, f)) and model(join(path, f)) is not None}

def store_agents(indices, matrix):
    """Store selected list of agents given by their keys"""
    agx, idx = [0]*len(indices), [0]*len(indices)
    for i, elem in enumerate(indices):
        agx[i] = matrix[elem]
        idx[i] = elem
    return agx, idx


### Generate Policy Matrix
def policy_matrix(agents):
    """Generates policy matrix (dictionary) of shape (n, n-1) for x agents"""
    matrix = {}
    for i, evaluation in enumerate(agents):
        matrix[evaluation] = []
        for j, behavior in enumerate(agents):
            if(i != j):
                matrix[evaluation].append(behavior)
    return matrix        

def get_behavior_policies(evaluation_policy):
    """Generates respective behvaior policies for a particular evaluation policy"""
    return policy_dict[evaluation_policy] if evaluation_policy in policy_dict.keys() else None

def get_trajectories(evaluation_agent, behavior_agent, N, type):
    """
    See formula above.
    Calculates the sum of value for a behavior policy and corresponding evaluation policy
    based on N trajectories.
    
    > Importance Sampling ratio, 𝜉
    > Compute sigma that represents inner sum for 1 behavior policy and corresponding evaluation policy
    
    @Param:
    1. evaluation_agent - (Agent) Evaluation Policy
    2. behavior_agent - (Agent) Behavior Policy
    3. N - (int) number of trajectories
    4. type - (int) 0/1 representing x_1 (0) and x_2 (1) for matrix multiplication later.
    @Return 
    - sigma = (float) returns inner sum, sigma (see formula for more details).
    """
    sigma = 0
    for _ in range(N):
        xi, reward = sampling_function(evaluation_agent, behavior_agent)
        sigma += reward * xi if(type == 0) else reward #X_2 doesn't use xi because it's distributed later.
    return sigma


def function(behavior_agents, evaluation_agent, N, type):
    """
    Computes f(π) using formula shown above.
    @Param:
    1. behavior_agent: (list) list of Agent class objects representing set of evaluation for given π_e.
    2. evaluation_agents: π_e (Agent) behavior policy
    3. N - (int) number of trajectories (used in calculation of sigma in `get_trajectories`)
    4. type: (int) 0/1 for calculation of X_1/X_2 with regards sum of products w.r.t constants, c and d.
    @Return
    - value: (float) value of function f(π) using formula above.
    """
    value = 0
    K = len(behavior_agents)
    for agent in behavior_agents:
        sigma = get_trajectories(evaluation_agent, agent, N, type)
        value += sigma
    
    return float(value / (K * N))

def Value(policy_dict, N):
    """
    Calculates the expected return for all evaluation agents
    @Param: 
    1. policy_dict - (dict[list]) policy matrix of shape (n, n-1) for n agents.
    2. N - (int) number of trajectories to run value estimation for.
    NOTE: N should equal with estimation value function parameter N.
    @Return:
    - values - (nd.array) Vector of values of evaluation policies using formula shown above. 
    """
    evaluation_agents = list(policy_dict.keys()) #generate n eval agents
    values = [0]*len(evaluation_agents) #vector of values of evaluation policies
    for i, agent in enumerate(evaluation_agents):
        value = [] #stores N Return for agent with policy π_e.
        for n in range(N):
            total_reward = 0
            state = env.reset() #reset
            counter = 0
            while True:
                action, prob = agent.get_action(agent.Q, state, eps=0)
                next_state, reward, done, info = env.step(action)
                total_reward += reward
                state = next_state
                counter += 1
                if(done or counter > 125): break
                    
            #append to value
            value.append(total_reward)
        #compute mean return and store in vector of values.
        expected_val = np.mean(value)
        values[i] = expected_val
    
    return np.array([values]).T #dim = number of agents.


#retrieve an evaluation policy with agent of index i.
get_eval_agent = lambda i: list(policy_dict.keys())[i - 1]


def main(policy_dict, N):
    """
    Main function for generating X_1 and X_2 of shapes k each, where k = number of base policies
    @Param:
    1. policy_dict - (dict[list]) policy matrix of shape (n, n-1) for n agents.
    2. N - (int) number of trajectories to run value estimation for.
    @Return:
    - X: (nd.array) concatenation of X_1 and X_2
    """
    X_1, X_2 = [], [] #store X_1 and X_2 for k policies
    K = len(policy_dict.keys())
    
    for i in range(1, K + 1):
        ### compute X_1 with evaluation policy, π_e = π_i
        evaluation_agent = get_eval_agent(i)
        ### generate set of behavior policies for π_e = π_1, i.e. π_k = {[π_j] for j ≠ i}
        behavior_agents = get_behavior_policies(evaluation_agent)
        x1 = function(behavior_agents, evaluation_agent, N, 0) #compute x1
        x2 = function(behavior_agents, evaluation_agent, N, 1) #compute x2
        ### store values
        X_1.append(x1)
        X_2.append(x2)
        
    #typecast to nd.array
    X_1 = np.array([X_1])
    X_2 = np.array([X_2])
    # ones = np.ones(X_1.shape)
    # X = np.hstack((X_1.T, X_2.T, ones.T)) #concat
    X = np.hstack((X_1.T, X_2.T)) #concat
    
    #Test dimensions
    assert(X.shape == (K, 2))
    return X

def test_model(params, policy_dict, type="lr"):
    """ Test algorithm """
    ### calculate X with 1k trajectories
    X = main(policy_dict, 1000)
    ### generate true value estimate
    y = Value(policy_dict, 1000)
    
    mse = MSE(type) #set regression algorithm
    mse.setParams(params)
    
    error = mse.mse(X, y, mode='test')
    
    return error

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run OPE for TD Learning')

    parser.add_argument('-agents', metavar='List_agents', type=str, default=[],
                        help='list of agents to store in policy matrix. Leave this blank to include all agents')
    parser.add_argument('-IS', metavar='str_sampling', type=str, default='OIS',
                        help='Enter type of Importance Sampling algorithm in OPE calculation.')
    parser.add_argument('-mse', metavar='str_mse_alg', default='lr',
                        help='Enter type of MSE algorithm for calculation of Value in MSE. Options = '\
                        'lr: linear regression; ridge: Ridge Regression; lasso: Lasso Regression; logit: Logistic Regression; maml: Model-Agnostic via Meta-Learning')

    args = parser.parse_args() #parse arguments

    env = utils.load_env() #Load environment
    agent = Agent(env) #create TD agent

    matrix = load_policies("../model/")
    

    """ Get arguments from user """
    agent_indices = utils.to_list(args.agents) if len(args.agents) > 0 else np.arange(1, len(matrix.keys()) + 1)
    
    agents, idx = store_agents(agent_indices, matrix) #pair agents
    
    #each key indicates evaluation policy, and corresponding values indicate behavior policies
    policy_dict = policy_matrix(agents)
    
    ### Importance Sampling
    sampling_function = IS(env, type=args.IS).func
    
    ### calculate X with 1k trajectories
    X = main(policy_dict, 1000)

    ### generate true value estimate
    true_values = Value(policy_dict, 1000)

    mse = MSE(args.mse) #set regression algorithm

    train_error = mse.mse(X, true_values) # train regression algorithm and compute the mean square error
    
    params  = mse.getParams() # get coefficients and y-intercept from running regression

    expected_return = true_values.mean() ## E[v(π)]

    ### Test Model with fitted params
    test_error = test_model(params, policy_dict, type=args.mse)
    
    
    
    
    # Display Stats
    print("********* Results *********")
    print("* Coefficients = ", params["coef_"])
    print("* Intercept = ", params["intercept_"][0])
    print("* Expected Return = ", expected_return)
    print("* Train MSE = ", train_error)
    print("* Test MSE: ", test_error)
    print("********* Results *********")