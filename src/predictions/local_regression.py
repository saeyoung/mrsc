import numpy as np

class LWRegressor():
    def __init__(self, f_type, params, fit_intercept=False):
        self.fit_intercept = fit_intercept
        self.f_type = f_type
        self.params = params
        
    def fit(X, y): 
        self.X = X
        self.y = y
        
    def predict(x0): 
        # add bias 
        if fit_intercept: 
            x0 = np.r_[1, x0]
            X = np.c_[np.ones(len(self.X)), self.X]
        else:
            X = self.X
        
        # fit model
        if self.f_type == 'inv_dist':
            w = self.inverse_distance(x0, X)
        else: 
            tau = self.params['tau']
            w = self.radial_kernel(x0, X, tau)
        xw = X.T * w
        beta = np.linalg.pinv(xw @ X) @ xw @ self.y
        
        # predict value
        return x0 @ beta
    
    def radial_kernel(self, x0, X, tau=0.001):
        return np.exp(np.sum((X - x0)**2 , axis=1) / (-2 * tau * tau))

    def inverse_distance(self, x0, X): 
        return 1 / np.sum((X - x0), axis=1)