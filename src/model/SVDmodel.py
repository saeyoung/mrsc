######################################################
#
# SVD Model
#
######################################################
import numpy as np
import pandas as pd
import random
import copy
import pickle
from sklearn import linear_model

import mrsc.src.utils as utils

class SVDmodel:
    def __init__(self, weights, singvals, target_data, donor_data, interv_index, total_index, setup, probObservation):
        """
        setup = [mat_form_method, denoise_method, denoise_mat_method, regression_method, skipNan]
        """
        self.weights = weights
        self.singvals = singvals
        self.target_data = target_data
        self.donor_data = donor_data
        self.num_k = len(weights)
        self.interv_index = interv_index
        self.total_index = total_index
        
        self.denoise_method = "SVD"
        self.denoise_mat_method = setup[2] # denoise_mat_method
        self.regression_method = setup[3] #regression_method
        self.skipNan = setup[4] # skipNan
        
        self.p = probObservation
        
        self.w_matrix = None
        # self.inv_w_matrix = None
        self.donor_pre = None
        self.target_pre = None
        self.beta = None

    def get_diagonal(self, weights, T):
        k = len(weights)
        diag_matrix = np.zeros((k*T, k*T))
        i = 0
        for weight in weights:
            rng = np.arange(i, i+T)
            diag_matrix[rng, rng] = weight
            i += T
        return diag_matrix
        
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

    def _prepare(self):
        # handle the nan's in target
        numOfNans = np.isnan(self.target_data).sum(axis=1).values / self.num_k
        if (numOfNans != 0):
            cols = self.target_data.columns[~np.isnan(self.target_data).values.flatten()]
            self.target_data = self.target_data.loc[:,cols]
            self.total_index = int(self.total_index - numOfNans)
            self.interv_index = int(self.total_index - numOfNans)
            
            if (self.skipNan == False):
                self.donor_data = self.donor_data.loc[:,cols]
            else:
                self.donor_data = utils.get_preint_data(self.donor_data, self.interv_index, self.total_index, self.num_k, reindex = True)
        # print("nan", numOfNans)
        # print("tar", self.target_data.columns)
        # print("tar", self.target_data.shape)
        # print("don", self.donor_data.columns)
        # print("don", self.donor_data.shape)

        # apply weights
        self.w_matrix = self.get_diagonal(self.weights, self.total_index)
        # self.inv_w_matrix = np.diag(1 / np.diag(self.w_matrix))

        # print("interv  ", self.interv_index)
        # print("total   ", self.total_index)
        # print("w_matrix", self.w_matrix.shape)
        # print("target  ", self.target_data.shape)
        # print("donor   ", self.donor_data.shape)
        self.target_data = self.target_data.dot(self.w_matrix)
        self.donor_data = self.donor_data.dot(self.w_matrix)

        # get self.donor_pre
        if (self.denoise_mat_method == "all"):
            df_hsvt = self.hsvt(self.donor_data, self.singvals)
            self.donor_pre = utils.get_preint_data(df_hsvt, self.interv_index, self.total_index, self.num_k)
        elif(self.denoise_mat_method == "pre"):
            df_pre = utils.get_preint_data(self.donor_data, self.interv_index, self.total_index, self.num_k)
            self.donor_pre = self.hsvt(df_pre, self.singvals)
        else:
            raise ValueError("Invalid denoise matrix method. Should be 'all' or 'pre'.")
            
        self.target_pre = utils.get_preint_data(self.target_data, self.interv_index, self.total_index, self.num_k)

    def fit(self):
        self._prepare()

        # regression
        if (self.regression_method == 'pinv'):
            self.beta = np.linalg.lstsq(self.donor_pre.T, self.target_pre.T, rcond=None)[0]
            # self.beta = np.linalg.pinv(self.donor_pre.T, rcond=1.0000000000000001e-14).dot(self.target_pre.T)

        elif (self.regression_method == 'lr'):
            regr = linear_model.LinearRegression(fit_intercept=True)
            regr.fit(self.donor_pre.T, self.target_pre.T)
            self.beta = regr.coef_
            self.beta = self.beta.T

        elif (self.regression_method == 'lasso'):
            regr = linear_model.Lasso(alpha = 1.0, fit_intercept=False)
            regr.fit(self.donor_pre.T, self.target_pre.T)
            self.beta = regr.coef_
            self.beta = self.beta.T
            
        else:
            raise ValueError("Invalid regression method. Should be 'lr' or 'pinv' or 'lasso'.")
        # print()
        # print(self.beta)
########################################################



