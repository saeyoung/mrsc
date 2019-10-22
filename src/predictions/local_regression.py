import numpy as np

class LWRegressor():
    def __init__(self, tau=1.0, fit_intercept=False):
        self.fit_intercept = fit_intercept
        self.tau = tau
        
    def fit(self, X, y): 
        self.X = X
        self.y = y
        
    def predict(self, x0): 
        # add bias 
        if self.fit_intercept: 
            x0 = np.r_[1, x0]
            X = np.c_[np.ones(len(self.X)), self.X]
        else:
            X = self.X
        
        # fit model
        w = self.radial_kernel(x0, X, self.tau)
        xw = X.T * w
        beta = np.linalg.pinv(xw @ X) @ xw @ self.y
        
        # predict value
        return x0 @ beta
    
    def radial_kernel(self, x0, X, tau=1.0):
        return np.exp(np.sum((X - x0) ** 2 , axis=1) / (-2 * tau * tau))

    def inverse_distance(self, x0, X): 
        return 1 / np.sum((X - x0) ** 2, axis=1)

"""class LWRegressor():
    def __init__(self, f_type, f_params, fit_intercept=False):
        self.fit_intercept = fit_intercept
        self.f_type = f_type
        self.f_params = f_params
        
    def fit(self, X, y): 
        self.X = X
        self.y = y
        
    def predict(self, x0): 
        # add bias 
        if self.fit_intercept: 
            x0 = np.r_[1, x0]
            X = np.c_[np.ones(len(self.X)), self.X]
        else:
            X = self.X
        
        # fit model
        if self.f_type == 'inv_dist':
            w = self.inverse_distance(x0, X)
        else: 
            tau = self.f_params['tau']
            w = self.radial_kernel(x0, X, tau)
        xw = X.T * w
        beta = np.linalg.pinv(xw @ X) @ xw @ self.y
        
        # predict value
        return x0 @ beta
    
    def radial_kernel(self, x0, X, tau=1.0):
        return np.exp(np.sum((X - x0) ** 2 , axis=1) / (-2 * tau * tau))

    def inverse_distance(self, x0, X): 
        return 1 / np.sum((X - x0) ** 2, axis=1)"""