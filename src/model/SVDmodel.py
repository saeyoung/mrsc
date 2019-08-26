######################################################
#
# SVD Model
#
######################################################
import numpy as np
import pandas as pd
import random
import copy

class SVDmodel:
    def __init__(self, singvals, target_data, donor_data, num_k, interv_index, total_index, denoise_mat_method, regression_method, probObservation):
        self.singvals = singvals
        self.target_data = target_data
        self.donor_data = donor_data
        self.num_k = num_k
        self.interv_index = interv_index
        self.total_index = total_index
        
        self.denoise_method = "SVD"
        self.denoise_mat_method = denoise_mat_method
        self.regression_method = regression_method 
        
        self.p = probObservation
        
        self.donor_pre = None
        self.target_pre = None
        self.beta = None
        
    def hsvt(self, df, rank): 
        """
        Input:
            df: matrix of interest
            rank: rank of output matrix
        Output:
            thresholded matrix
        """
        if (rank == 0 | rank > min(df.shape)):
            return df
        u, s, v = np.linalg.svd(df, full_matrices=False)
        s[rank:].fill(0)
        vals = (np.dot(u*s, v))
        return pd.DataFrame(vals, index = df.index, columns = df.columns)
    
    def get_preint_data(self, combinedDF, intervIndex, totalIndex, nbrMetrics, reindex = True):
        """
        Input:
            combinedDF: (dataframe) concatenated df of size (N, nbrMetrics*totalIndex)
            intervIndex: pre-int period
            totalIndex: total period
            nbrMetrics: number of metrics

        Output:
            pre intervention of all metrics, concatenated
        """
        if reindex:
            combinedDF.columns = range(combinedDF.shape[1])
        indexToChoose = []
        for k in range(nbrMetrics):
            indexToChoose = indexToChoose + list(range(k*totalIndex,k*totalIndex + intervIndex))
        return combinedDF.loc[:,indexToChoose]

    def _prepare(self):        
        # get self.donor_pre
        if (self.denoise_mat_method == "all"):
            df_hsvt = self.hsvt(self.donor_data, self.singvals)
            self.donor_pre = self.get_preint_data(df_hsvt, self.interv_index, self.total_index, self.num_k)
        elif(self.denoise_mat_method == "pre"):
            df_pre = self.get_preint_data(self.donor_data, self.interv_index, self.total_index, self.num_k)
            self.donor_pre = hsvt(df_pre, self.singvals)
        else:
            raise ValueError("Invalid denoise matrix method. Should be 'all' or 'pre'.")
            
        # get self.target_pre
        self.target_pre = self.get_preint_data(self.target_data, self.interv_index, self.total_index, self.num_k)
    
    def fit(self):
        self._prepare()
        
        if (self.regression_method == 'lr'):
            regr = linear_model.LinearRegression(fit_intercept=True)
            regr.fit(self.donor_pre.T, self.target_pre.T)
            self.beta = regr.coef_
            
        elif (self.regression_method == 'pinv'):
            self.beta = np.linalg.pinv(self.donor_pre.T).dot(self.target_pre.T)
        
        elif (self.regression_method == 'lasso'):
            regr = linear_model.Lasso(alpha = 1.0, fit_intercept=False)
            regr.fit(self.donor_pre.T, self.target_pre.T)
            self.beta = regr.coef_
            
        else:
            raise ValueError("Invalid regression method. Should be 'lr' or 'pinv' or 'lasso'.")
########################################################



