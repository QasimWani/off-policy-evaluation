###### Requires python3.8+
### Author: Qasim Wani
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gym
import time
import tqdm
from collections import defaultdict
import sys

from model import Agent
import utils

#create gym environment
env_name = "Taxi-v3"
env = gym.make(env_name)
env = env.unwrapped #gets ride of TimeLimit

def train(epochs, path):
    """ Train QLearning agent """
    epochs = int(epochs)
    
    agent = Agent(env) #create TD agent
    # obtain the estimated optimal policy and corresponding action-value function
    Q_optimal, policy_optimal = utils.TD_learning(agent, epochs, 0.1, td_type="Q")
    reward_optimal, steps_optimal = utils.run_optimal_policy(agent, Q_optimal, 1000, td_type="Q", plot=True)
    print(f"Average reward at the end of {epochs} episodes: {reward_optimal.mean()}")

    utils.save_policy(agent, "../model/" + str(path) + ".npy") #save policy

if __name__ == "__main__":
    path, epochs = "x", 50e3 #default settings
    if(len(sys.argv) == 3):
        path, epochs = sys.argv[1:]
    
    train(epochs, path)
    
    
        