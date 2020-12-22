### functions for running TD algorithms

from model import Agent
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import sys
import pickle


def TD_learning(agent, num_episodes, alpha, gamma=1.0, td_type = "sarsa"):
    """
    Runs the optimal TD (specified in params) operation and returns the optimal policy and Q table.
    Utlizes the agent class created previously.
    @Params:
    1. agent: Instance of agent class.
    2. num_episodes: number of episodes to train for.
    3. alpha: value of alpha for regularizer.
    4. gamma: discounted return, Ɣ (refer to Bellman Equations)
    5. td_type: type of TD control. Choose str between 'sarsa', 'Q', or 'expected_sarsa'
    @Returns:
    1. Q - estimated optimal Q table.
    2. policy - estimated optimal policy, π.
    """
    # initialize performance monitor
    steps_arr = []
    reward_arr = []
    # loop over episodes
    for i_episode in range(1, num_episodes+1):
        # monitor progress
        if i_episode % 100 == 0:
            print("\rEpisode {}/{}".format(i_episode, num_episodes), end="")
            sys.stdout.flush()   
        
        ## Q-learning implementation

        epsilon = 1.0/i_episode #impose GLIE conditions
        Q_updated, steps, reward = agent.generate_episode((gamma, alpha, epsilon), TD_type = td_type)
        Q = Q_updated #update value for Q-table
        #performance monitor
        steps_arr.append(steps)#append #steps for current episode (should decrease as num_episode -> ∞)
        reward_arr.append(reward)#should approach optimal reward as num_episode -> ∞
    
    #convert performance metrics to nd.array
    steps_arr = np.array(steps_arr)
    reward_arr = np.array(reward_arr)
    plt.plot(steps_arr, label="steps")#plot steps
    plt.plot(reward_arr, label="reward")#plot reward
    plt.xlabel("Number of episodes")
    plt.ylabel("Continuous value")
    plt.ylim(-300, 300)#set y-axis boundaries
    plt.legend(loc="upper right")#display legend
    plt.show()
    
    print("SUMMARY")
    print(f"\nAverage Reward: {reward_arr.mean()}\nAverage Steps to completion: {steps_arr.mean()}")
    print("\n*******BEFORE*******")
    print(f"\nMinimum Reward attained at the end of an episode: {reward_arr.min()} at pos: {reward_arr.argmin()}")
    print(f"Greatest number of steps for an episode: {steps_arr.max()} at position: {steps_arr.argmax()}")
    print("\n*******AFTER*******")
    print(f"Maximum Reward attained at an episode: {reward_arr.max()} at position #: {reward_arr.argmax()}")
    print(f"Fewest number of steps took for task completion: {steps_arr.min()} at pos: {steps_arr.argmin()}")
    
    ### Find the optimal policy, π*
    policy = np.array([np.argmax(Q[key]) if key in Q else -1 for key in np.arange(500)]).reshape((5, 5, 5, 4))
    return Q, policy



def run_optimal_policy(agent, Q_optimal, num_episodes, td_type = 'sarsa', epsilon=0.0, gamma=1.0, alpha=1.0, plot=False):
    """
    Runs the optimal policy for an agent.
    @Params:
    1. agent: Instance of Agent class.
    2. Q_optimal: optimal Q-table (must be trained above)
    3. num_episodes: number of episodes to test Q-table for.
    4. td_type: TD control type. By default, 'sarsa'
    5. epsilon: rate of exploration (used in greedy policy) hyper-parameter.
    6. gamma: discounted return regularizer hyper-parameter.
    7. alpha: control predictive regularizer hyper-parameter.
    @Returns:
    1. average_reward at the end of running num_episodes.
    2. average_steps for completion.
    3. matplotlib.pyplot chart for reward (pprint only)
    """
    agent.Q = Q_optimal
    values = (gamma, alpha, epsilon)
    steps_arr = [] #initialize steps counter
    reward_arr = [] #initializer reward counter
    for i in range(1, num_episodes + 1):
        # monitor progress
        if i % 100 == 0:
            print("\rEpisode {}/{}".format(i, num_episodes), end="")
            sys.stdout.flush()
        
        _, steps, reward = agent.generate_episode(values, TD_type = td_type)
        #performance monitor
        steps_arr.append(steps)#append #steps for current episode (should decrease as num_episode -> ∞)
        reward_arr.append(reward)#should approach optimal reward as num_episode -> ∞
    
    #convert performance metrics to nd.array
    steps_arr = np.array(steps_arr)
    reward_arr = np.array(reward_arr)
    
    if(plot):#plot results
        plt.plot(steps_arr, label="steps")#plot steps
        plt.plot(reward_arr, label="reward")#plot reward
        plt.xlabel("Number of episodes")
        plt.ylabel("Continuous value")
        plt.ylim(-30, 20)#set y-axis boundaries
        plt.legend(loc="upper right")#display legend
        plt.show()
    
    return reward_arr, steps_arr

def save_policy(agent, path):
    """Saves a policy of an agent"""
    np.save(path, dict(agent.Q))
    print("Saved policy!")
    

def load_policy(path, nA=6):
    """Load a policy"""
    policy = np.load(path, allow_pickle=True)
    return defaultdict(lambda : np.zeros(nA), policy.all())