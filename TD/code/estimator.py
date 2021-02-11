### define Importance Sampling Estimator functions
import numpy as np

from model import Agent
from maml import Importance_Sampling

class IS():
    """Class to implement different Importance Sampling functions"""
    def __init__(self, env, isMAML, type="OIS"):
        """Define Importance Sampling function type"""
        self.env = env
        self.func = self.RIS if(type == "RIS") else self.OIS
        self.isMAML = isMAML
        self.D = None #build dataset for computing π_d

    def OIS(self, evaluation_agent:Agent, behavior_agent:Agent):
        """
        > Calculates ordinary importance sampling between behavior agent and evaluation agent
        > Calculates Return value (total reward) for behavior_agent
        NOTE: behavior_agent (π_k) generates all states and action.
        @Param:
        1. evaluation_agent - agent class object representing evaluation policy.
        2. behavior_agent - agent class object representing behavior policy.
        @Returns:
        - prod : importance sampling ratio b/w evaluation and behavior agent.
        - var  : importance sampling variance 
        - total_reward : return of behavior_agent.
        """
        total_reward = 0
        state = self.env.reset() #reset

        counter = 0 #counter for maximum limit

        OIS = {'pi_e': [], 'pi_k': []}
        ois = []
        while True:
            action, prob_behv = behavior_agent.get_action(behavior_agent.Q, state, eps=0) #generate best action and prob for behavior.
            _, prob_eval = evaluation_agent.get_action(evaluation_agent.Q, state, eps=0) #generate max probability for evaluation policy.
            
            #store π(s|a) for behavior and evaluation policy
            OIS['pi_e'].append(prob_eval); OIS['pi_k'].append(prob_behv)
            ois.append( float(prob_eval/prob_behv) )
            
            next_state, reward, done, info = self.env.step(action) #transition

            total_reward += reward #update reward
            state = next_state #update state
            counter += 1
            
            if(done or counter > 125): #stopping condition
                break

        if self.isMAML:
            is_maml = Importance_Sampling(OIS['pi_k'], OIS['pi_e']) #define MAML agent
            is_maml.update() #perform single step MAML update
            ois = is_maml.seekParams() #get updated Importance Sampling weights.
        
        return np.prod(ois), np.var(ois * total_reward), total_reward
    
    def RIS(self, evaluation_agent, behavior_agent):
        """
        > Calculates Regression Importance Sampling ratio between evaluation and behavior policy.
        > Calculates Return (total reward) for behavior_agent
        NOTE: behavior_agent (π_k) generates all states and action.
        @Param:
        1. evaluation_agent - agent class object representing evaluation policy.
        2. behavior_agent - agent class object representing behavior policy.
        @Returns:
        - prod : importance sampling ratio b/w evaluation and behavior agent.
        - var  : importance sampling variance 
        - total_reward : return of behavior_agent.
        """
        #Steps: 
        # 1. generate feature matrix, D, conditioned on π_b DONE
        # 2. count(h, a) and count(h) accordingly. 
        # 3. generate the estimate for π_b as π_d
        # 4. compute the RIS ratio and return value
        D, reward = self.gen_feat_matrix(behavior_agent)
        self.D = D #set dataset

        RIS = {'pi_e': [], 'pi_k': []}
        ris = []
        
        for state, action in zip(*D):
            _, prob_eval = evaluation_agent.get_action(evaluation_agent.Q, state, eps=0) #generate max probability for evaluation policy.
            prob_estimate_behv = self.count(state, action)
            #store π(s|a) for behavior and evaluation policy
            RIS['pi_e'].append(prob_eval); RIS['pi_k'].append(prob_estimate_behv)
            ris.append( float(prob_eval/prob_estimate_behv) )
            
        if self.isMAML:
            is_maml = Importance_Sampling(RIS['pi_k'], RIS['pi_e']) #define MAML agent
            is_maml.update() #perform single step MAML update
            
            ris = is_maml.seekParams() #get updated Importance Sampling weights.
        
        return np.prod(ris), np.var(ris * reward), reward
    
    def count(self, observation, action):
        """Compute count-based estimate for π_d(a|h_{i - n:i})"""
        if(self.D is None):
            raise ValueError("Dataset cannot be empty!")

        count_action, count_observation = 0, 0 #define count(h, a) and count(h) respectively
        for s, a in zip(*self.D):
            if(observation == s and action == a):#update count(h, a)
                count_action += 1
                
            if(observation == s): #update count(h)
                count_observation += 1
        
        return count_action / count_observation
                
            
        
    def gen_feat_matrix(self, agent):
        """Generate data D and associated return conditioned on behavior policy π_b"""
        if(self.func != self.RIS):
            raise ValueError("gen_feat_matrix requires using RIS as sampling function")
        
        S = [] #set of States in Dataset, D
        A = [] #set of actions conditioned on S^{n - 1} in dataset, D generated by π_b
        
        state = self.env.reset()
        
        counter = 0
        total_reward = 0
        while True:
            action, _ = agent.get_action(agent.Q, state, eps=0) #generate best action and prob for behavior.
            next_state, reward, done, info = self.env.step(action) #transition

            #### Update ####
            counter += 1
            S.append(state)
            A.append(action)
            #### Update ####
            
            total_reward += reward #update reward
            state = next_state #update state        
            
            if(done or counter > 125): #stopping condition
                break
        return (S, A), total_reward