#implementation of MAML for updating coefficients of regression task.

import torch
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
from torch.autograd import Variable

import numpy as np

from model import Agent
from estimator import IS

class MAML():
    """ Implementation of Model Agnostic meta-learning algorithm """
    def __init__(self, X, values, param_size:tuple, alpha:float=0.01, beta:float=0.05, theta=None, bias:float=None):
        """ 
            Initialize params and agents for tasks as defined:
            1. X: (nd.array) array of shape (N, 3) where N represents the number of evaluation agents used to calculate the gradient updates from policy_dict (see ope.py).
            2. values: (nd.array) array of shape (N, 1) where N represents the number of evaluation agents to represent the expected return for N agents, i.e. E[v(π)].
            3. param_size: (tuple(int, int)) total number of parameters to initialize theta with. Excluding bias/y-intercept.
            4. alpha: (float) internal gradient update hyperparameter. See MAML paper for details.
            5. beta: (float) meta-update hyperparameter. See MAML paper for details.
            6. theta: (nd.array[np.float32]) optional param to perform meta-update on input parameter.
            7. bias: (float) y-intercept for parameters theta.
        """
        #enable cuda if possible
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        #parameter distribution
        self.coef_ = torch.tensor(theta, requires_grad=True) if theta is not None else torch.rand(param_size, requires_grad=True)
        self.intercept_ = torch.tensor(bias, requires_grad=True) if bias is not None else torch.rand(1, requires_grad=True) #get y-intercept
        
        #define tasks and associated return values
        self.tasks = X
        self.y = torch.FloatTensor(values)

        #define MAML hyperparameters
        self.alpha = alpha
        self.beta = beta
        
        self.criteon = nn.MSELoss() #define loss function
        
        self.meta_optim = optim.Adam([self.coef_, self.intercept_], lr=self.beta)
    
    def update(self, max_norm=5.0):
        """ Run a single iteration of MAML algorithm """
        # wip w/o batch sampling
        theta_prime = []
        bias_prime = []
        for i, task in enumerate(self.tasks):
            y_hat = self.f( self.coef_, self.intercept_, task ) #task specific reward prediction
            loss = self.criteon( y_hat, self.y)
            #compute gradients
            grad = torch.autograd.grad(loss, [self.coef_, self.intercept_])
            #update params
            theta_prime.append( self.coef_ - self.alpha * grad[0] )
            bias_prime.append( self.intercept_ - self.alpha * grad[1] )

        del loss
        
        #perform meta-update
        m_loss = torch.tensor(0.0, requires_grad=True)
        for i in range(len(self.tasks)):
            theta = theta_prime[i]
            bias = bias_prime[i]
            task = self.tasks[i]
            y_hat = self.f( theta, bias, task ) #task specific reward prediction
            m_loss = m_loss + self.criteon( y_hat, self.y)
 
        #zero gradient before running backward pass
        self.meta_optim.zero_grad()

        #backward pass
        m_loss.backward(retain_graph=True)

        #one-step gradient descent
        nn.utils.clip_grad_norm_([self.coef_, self.intercept_], max_norm) #clip gradients
        
        self.meta_optim.step()
        
        return m_loss.data
    
    def f(self, theta, bias, X):
        """ Compute dot product of X w.r.t parameters theta """
        X = torch.FloatTensor(X).reshape(3, 1)
        dot = torch.matmul( theta, X) # (N x 3) • (3 x 1)
        dot.requires_grad_() #bug fix to retain computational graph
        return dot + bias # shape = (N, 1)
    
    def predict(self, X):
        """ compute y_hat based on maml trained parameters """
        X = torch.FloatTensor(X.T)
        theta = self.coef_.clone()
        dot = torch.matmul(theta, X).detach().data
        dot = torch.mean(dot, 1, True)

        return dot.numpy() + self.intercept_.detach().numpy()
            