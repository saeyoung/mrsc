import numpy as np
from sklearn.metrics.pairwise import pairwise_kernels
#cosine_similarity, sigmoid_kernel, rbf_kernel, polynomial_kernel

class LWRegressor():
    def __init__(self, kernel='rbf', alpha=0.0, params=None, fit_intercept=False):
        self.fit_intercept = fit_intercept
        self.kernel = kernel
        self.alpha = alpha 
        self.params = params
        self.coef_ = np.array([])
        
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
        x0_input = x0.reshape(1, x0.shape[0])
        w = pairwise_kernels(X, x0_input, self.kernel).flatten()
        xw = X.T * w
        self.coef_ = np.linalg.pinv(xw @ X + self.alpha * np.eye(X.shape[1])) @ xw @ self.y

        # predict value
        return x0 @ self.coef_

        """if kernel == 'cosine': 
                                    w = cosine_similarity(X, x0_input)
                                elif kernel == 'sigmoid':
                                    gamma = self.params['gamma']
                                    w = sigmoid_kernel(X, x0, gamma)
                                elif kernel == 'poly':
                        
                                    w = polynomial_kernel(X, x0, )
                                elif kernel == 'laplace':
                        
                                else: 
                                    gamma = self.params['gamma']
                                    w = rbf_kernel(X, x0, gamma)"""
    
    """def radial_kernel(self, x0, X):
                    return np.exp( - self.gamma * np.sum((X - x0) ** 2 , axis=1) )
            
                def rbf_kernel(self,m x0, X):
                    return rbf_kernel
            
                def cosine_kernel(self, x0, X): 
                    return cosine_similarity(X, x0.reshape(1, x0.shape[0]))
            
                def sigmoid_kernel(self, x0, X): 
                    return sigmoid_kernel(X, x0.reshape(1, x0.shape[0]), self.gamma)
            
                    #return np.exp(np.sum((X - x0) ** 2 , axis=1) / (-2 * tau * tau))
            
                def inverse_distance(self, x0, X): 
                    return 1 / np.sum((X - x0) ** 2, axis=1)"""

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