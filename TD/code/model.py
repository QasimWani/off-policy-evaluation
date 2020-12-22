### Implement TD control to solve OpenAI's Taxi problem using gym.
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import time
import sys

#create gym environment
# env_name = "Taxi-v3"
# env = gym.make(env_name)
# env = env.unwrapped #gets ride of TimeLimit


#Agent class
class Agent():
    """
    This class is used in finding/converging towards the optimal policy for Taxi-lab environment in OpenAI.
    User can select between either of the 3 TD control methods, i.e. sarsa, sarsa_max (Q), or expected_sarsa.
    
    Passenger locations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
    - 4: in taxi
    
    Destinations:
    - 0: R(ed)
    - 1: G(reen)
    - 2: Y(ellow)
    - 3: B(lue)
        
    Actions:
    There are 6 discrete deterministic actions:
    - 0: move south
    - 1: move north
    - 2: move east 
    - 3: move west 
    - 4: pickup passenger
    - 5: dropoff passenger
    
    -----
    Read individual method docstring for more info.
    Resources:
    1. Taxi Environment paper: https://arxiv.org/pdf/cs/9905014.pdf
    2. OpenAI simulation: https://gym.openai.com/envs/Taxi-v3
    -----
    Author: Qasim Wani
    Date written: December 21, 2020.
    """
    def __init__(self, env):
        self.nA = env.action_space.n
        self.Q = defaultdict(lambda : np.zeros(self.nA))
        self.env = env
        
    def encode(self, row, col, loc, dest):
        """
        Given a tuple of observation consisting of taxi's position (row, column)
        with the passengers current location and point of destination, encode it
        into an integer of range: (0, 500].
        @Param:
        1. row: Taxi row number: (int) 0 - 4
        2. col: Taxi column number: (int) 0 - 4
        3. loc: passenger location: one of the 4 colors, Red, Green, Blue, Yellow, or in-taxi (full)
        4. dest: passenger destination: one of the 4 corners in the grid indicated as: R/G/B/Y
        @Returns:
        - i: encoded number given the observational state of the environment, env.
        Credits: https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py#L130
        """
        i = row
        i *= 5
        i += col
        i *= 5
        i += loc
        i *= 4
        i += dest
        return i
    
    def decode(self, i):
        """
        Given an integer (0, 500], decode it into the following:
        1. taxi row.
        2. taxi column.
        3. passenger location.
        4. passenger destination.
        Credits: https://github.com/openai/gym/blob/master/gym/envs/toy_text/taxi.py#L141
        """
        out = []
        out.append(i % 4)
        i = i // 4
        out.append(i % 5)
        i = i // 5
        out.append(i % 5)
        i = i // 5
        out.append(i)
        return tuple(reversed(out))
    
    def initialize_env(self, decode=True):
        """
        Initializes the environment, randomly, and decodes the observational state using decode() f(x).
        Uses env.reset() internally.
        Return the observational state as a decoded version if decode is set to true (default)
        Otherwise, returns the encoded state version.
        """
        return self.decode(self.env.reset()) if decode else self.env.reset()
    
    def generate_episode(self, values, TD_type = "sarsa"):
        """
        Generates an episode & updates the Q_table along the way.
        Parameters:
        1. values: tuple of gamma, alpha, & epsilon values.
        2. TD_type: type of TD control to use. default = 'sarsa'. Options: "sarsa", "Q", "expected_sarsa".
        Returns:
        - Q : Updated Q table.
        - steps_to_completion: #steps it took to complete an episode.
        - total_reward: total reward accumulated during the episode.
        """
        state = self.initialize_env(decode = False) #initial state, decoded
        gamma, alpha, epsilon = values #extract information
        nA = self.nA #action space size.
        steps_to_completion:int = 0 #number of steps taken to task completion in one episode
        total_reward:int = 0 #total reward accumulated in an episode
        Q = self.Q
        while True:
            action = self.get_action(Q, state, epsilon)#choose between greedy or equiprobable action

            next_state, reward, done,_ = self.env.step(action) #step into the episode
            value = Q[state][action] #get current value
            next_action = self.get_action(Q, next_state, epsilon)#choose between greedy/random for next state

            # Choose b/w 3 TD control algorithms.
            #SARSA: Q(S0, A0) --> Q(S0, A0) + å(R1 + gamma(Q(S1, A1) - Q(S0, A0)))
            #SARSA Max: Q(S0, A0) --> Q(S0, A0) + å(R1 + gamma(max (Q(S1, a)) - Q(S0, A0)))
            #Expected SARSA: Q(S0, A0) --> Q(S0, A0) + å(R1 + gamma(Σ[ π(a | S1)* Q(S1, a) ] - Q(S0, A0)))

            target = self.get_value_TD(next_state, next_action, TD_type, epsilon) #choice of TD
            value += alpha*(reward + gamma*(target - value))
            
            Q[state][action] = value #update Q-table
                
            state = next_state #update state
            steps_to_completion += 1 #increment number of steps to episode completion
            total_reward += reward #increment reward counter

            if(done): #episode completed
                break
        return Q, steps_to_completion, total_reward

    def get_action(self, Q, state, eps):
        """
        Gets the action following epsilon greedy policy.
        Parameters:
        1. Q: Q-table.
        2. state: current state in the episode.
        3. eps: epsilon value following GLIE conditions.
        Returns:
        - actions: the action to take based on choosing greedy action or random action
        """
        #exploit if True else explore
        return np.argmax(Q[state]) if np.random.random() > eps else np.random.choice(np.arange(self.nA))
    
    def get_value_TD(self, next_state, next_action = None, TD_type = "sarsa", epsilon = None):
        """
        Returns the value from the Q table based on the type of TD control algorithm
        @Param:
        1. next_state: S_(t+1) from the episode
        2. next_action: A_(t+1). None if TD_type ≠ 'sarsa'
        3. TD_type: type of TD control algorithm to use. Can be 'sarsa', 'Q', or 'expected_sarsa'
        4. epsilon: value for epsilon.
        Returns:
        - value: value from the Q-table based on the Temporal Difference control algorithm used.
        """
        Q = self.Q #retrieve the current Q table
        if(TD_type == "sarsa"):
            return Q[next_state][next_action] if next_action is not None else 0
        elif(TD_type == "Q"):
            return np.max(Q[next_state])
        elif(TD_type == "expected_sarsa"):
            policy_s = np.ones(self.nA) * epsilon / self.nA
            policy_s[np.argmax(Q[next_state])] = 1 - epsilon + (epsilon / self.nA)
            return np.dot(Q[next_state], policy_s)
        else:
            raise ValueError("Invalid TD_type specified")
