### provide wrapper for regression function for MSE calculation

######################
## MSE Calculation ##
# 1. Linear Regression
# 2. Ridge Regression
# 3. Lasso Regression
# 4. Logistic Regression (should binning be used to convert continuous variable to categorical?)
# 5. meta-learning
#####################
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn import metrics
import numpy as np

from maml import MAML

class MSE():
    """Perform OPE caculation using MSE as loss function"""
    def __init__(self, type, param_size=None):
        """
        Specify type of regression algorithm for wrapper on MSE
        >>> type: (str) "lr" [simple linear regression], "ridge" [ridge regression],..., "maml" [meta-learning]
        >>> param_size: (tuple) meta-learning parameter representing shape of policy_dict (see ope.py for more info on policy_dict)
        """
        self.model = None
        self.degree = 3 #default polynomial order
        self.param_size = param_size #maml param

        if(type == "lr"):
            self.alg = self.linear_regression
            self.model = LinearRegression

        elif(type == "ridge"):
            self.alg = self.ridge_regression
            self.model = Ridge
        
        elif(type == "lasso"):
            self.alg = self.lasso_regression
            self.model = Lasso
       
        elif(type == "logit"):
            self.alg = self.logistic_regression
            
        elif(type == "poly"):
            self.alg = self.polynomial_regression
            self.model = LinearRegression
        
        elif(type == "maml"):
            self.alg = self.maml
            self.model = LinearRegression
        
        else:
            raise ValueError("Incorrect type specificed. See docs for `type`")
        
        self.coef = self.intercept = None
        self.reg = None
        
    def predict(self, X):
        """ generate y_hat from test observations, X """
        if(self.reg is None and self.model is not None):
            self.reg = self.model()
            params = self.getParams()
            self.reg.intercept_ = params["intercept_"]
            self.reg.coef_ = params["coef_"]

        #check for polynomial model
        if(self.alg == self.polynomial_regression):
            X = PolynomialFeatures(degree=self.degree).fit_transform(X)
        
        return self.reg.predict(X)
    
    def linear_regression(self, X, y):
        """transform & fit linear regression model to feature matrix, X"""
        
        self.reg = LinearRegression().fit(X, y) 
        if(self.coef is None):
            self.coef = self.reg.coef_
            self.intercept = self.reg.intercept_
        else:
            self.reg.coef_ = self.coef
            self.reg.intercept_ = self.intercept

        return self.reg.predict(X)
    
    def logistic_regression(self, X, y):
        """transform & fit logistic regression model to feature matrix, X"""
        raise NotImplementedError
    
    def ridge_regression(self, X, y):
        """transform and fit ridge regression model to feature matrix, X for ridge regression"""
        self.reg = Ridge().fit(X, y) 
        if(self.coef is None):
            self.coef = self.reg.coef_
            self.intercept = self.reg.intercept_
        else:
            self.reg.coef_ = self.coef
            self.reg.intercept_ = self.intercept
        
        return self.reg.predict(X)
    
    def polynomial_regression(self, X, y):
        """transform and fit polynomial regression model with degree 2 to feature matrix, X"""
    
        X = PolynomialFeatures(degree=self.degree).fit_transform(X)
        
        self.reg = LinearRegression().fit(X,y)  
        if(self.coef is None):
            self.coef = self.reg.coef_
            self.intercept = self.reg.intercept_
        else:
            self.reg.coef_ = self.coef
            self.reg.intercept_ = self.intercept

        return self.reg.predict(X)
    
    def lasso_regression(self, X, y):
        """transform and fit lasso regression model to feature matrix, X for lasso regression"""
        
        self.reg = Lasso().fit(X, y) 
        if(self.coef is None):
            self.coef = self.reg.coef_
            self.intercept = self.reg.intercept_
        else:
            self.reg.coef_ = self.coef
            self.reg.intercept_ = self.intercept

        return self.reg.predict(X)
        
    def maml(self, X, y):
        """implementation of model-agnostic meta-learning for parameter generation"""
        self.reg = MAML(X, y, self.param_size)
        self.reg.update() #perform 1 gradient update TODO: replace original/default params with trained params from another regression alg.
        if(self.coef is None):
            self.coef = self.reg.coef_.detach().numpy()
            self.intercept = self.reg.intercept_.detach().numpy()
        else:
            self.reg.coef_.data = self.coef
            self.reg.intercept_.data = self.intercept
            
        return self.reg.predict(X)
        
    def mse(self, X, y, mode="train"):
        """Calculates MSE"""
        return metrics.mean_squared_error(self.alg(X, y) if mode == "train" else self.predict(X), y)
    
    def getParams(self):
        """Gets the parameters from algorithm"""
        return {"coef_" : self.coef, "intercept_" : self.intercept}
    
    def setParams(self, params):
        """Sets the paramer for algorithm"""
        self.intercept = params["intercept_"]
        self.coef = params["coef_"]
        print("Set params!")