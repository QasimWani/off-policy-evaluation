### provide wrapper for regression function for MSE calculation

######################
## MSE Calculation ##
# 1. Linear Regression
# 2. Ridge Regression
# 3. Lasso Regression
# 4. Logistic Regression (should binning be used to convert continuous variable to categorical?)
# 5. meta-learning
#####################

class MSE():
    """Perform OPE caculation using MSE as loss function"""
    def __init__(self, type):
        """
        Specify type of regression algorithm for wrapper on MSE
        >>> type: (str) "lr" [simple linear regression], "ridge" [ridge regression],..., "maml" [meta-learning]
        """
        if(type == "lr"):
            self.alg = self.linear_regression
        elif(type == "ridge"):
            self.alg = self.ridge_regression
        elif(type == "maml"):
            self.alg = self.maml
        elif(type == "lasso"):
            self.alg = self.lasso_regression
        elif(type == "logit"):
            self.alg = self.logistic_regression
        else:
            raise ValueError("Incorrect type specificed. See docs for `type`")
        
        self.coef = self.intercept = None
        
    def linear_regression(self, X, y):
        """transform & fit linear regression model to feature matrix, X"""
        from sklearn.linear_model import LinearRegression
        reg = LinearRegression().fit(X, y) 
        
        if(self.coef is None):
            self.coef = reg.coef_
            self.intercept = reg.intercept_
        else:
            reg.coef_ = self.coef
            reg.intercept_ = self.intercept

        return reg.predict(X)
    
    def logistic_regression(self, X, y):
        """transform & fit logistic regression model to feature matrix, X"""
        raise NotImplementedError
    
    def ridge_regression(self, X, y):
        """transform and fit ridge regression model to feature matrix, X for ridge regression"""
        from sklearn.linear_model import Ridge
        reg = Ridge().fit(X, y) 
        if(self.coef is None):
            self.coef = reg.coef_
            self.intercept = reg.intercept_
        else:
            reg.coef_ = self.coef
            reg.intercept_ = self.intercept

        return reg.predict(X)
    
    def lasso_regression(self, X, y):
        """transform and fit lasso regression model to feature matrix, X for lasso regression"""
        from sklearn.linear_model import Lasso
        reg = Lasso().fit(X, y) 
        if(self.coef is None):
            self.coef = reg.coef_
            self.intercept = reg.intercept_
        else:
            reg.coef_ = self.coef
            reg.intercept_ = self.intercept

        return reg.predict(X)
        
    def maml(self, X, y):
        """implementation of model-agnostic meta-learning for parameter generation"""
        raise NotImplementedError
        
    def mse(self, X, y):
        """Calculates MSE"""
        from sklearn import metrics
        return metrics.mean_squared_error(self.alg(X, y), y)
    
    def getParams(self):
        """Gets the parameters from algorithm"""
        return {"coef_" : self.coef, "intercept_" : self.intercept}
    
    def setParams(self, params):
        """Sets the paramer for algorithm"""
        self.intercept = params["intercept_"]
        self.coef = params["coef_"]
        print("Set params!")