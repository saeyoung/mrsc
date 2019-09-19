################################################################
#
# MultiDimensional Robust Synthetic Control (mRSC)
#
################################################################
import numpy as np
import pandas as pd
import copy
from matplotlib import pyplot as plt

from mrsc.src.model.SVDmodel import SVDmodel
from mrsc.src.model.Target import Target
from mrsc.src.model.Donor import Donor

import mrsc.src.utils as utils

class mRSC:
    def __init__(self, donor, target, pred_interval=1, probObservation=1.): 
        """
        donor = (Donor)
        target = (Target)
        probObservation = (float) a value between 0 and 1

        
        """
        self.donor = donor
        self.target = target
        self.p = probObservation
        
        """ 
        interv_index = (int) intervention index to split trian/test (T_0)
        total_index = (int) total size of the full timeline (T)
        pred_interval = (int) prediction interval (T - T_0)

        metrics = (list) list of metrics in donor/target matrix
        num_k = (int) number of metrics
        weighting = (string) how we weight the dataset (None, "normalize")

        mat_form_method = (string) 'fixed' or 'sliding'
        skipNan = (boolean) True if we skip the nan in the data (dormant year does not count in)
                            False if we keep the target's nan and construct donor matrix,
                                  and then remove the NaN columns after.
        nan_index = (list) the column index of the NaN values in target_data
        target_data = (df) target_matrix
        donor_data = (df) donor matrix
        
        model = SVD or ALS model
        """

        # time indices
        self.pred_interval = pred_interval
        self.total_index = self.target.total_index
        self.interv_index = self.total_index - self.pred_interval

        # metrics related
        self.metrics = None
        self.num_k = None
        self.weighting = None

        # matrix formulation
        self.mat_form_method = None
        self.skipNan = None
        self.nan_index = None
        self.target_data = None
        self.donor_data = None
        self.weights = None

        # denoise model
        self.model = None
        # self.model.beta has the weights learned

        # normalize the columns of donor_data and target_data
        # only use donor_data to get mean and var
    def normalize_col(self, mean_list, var_list):
        self.weights = [mean_list, var_list]
        self.donor_data = (self.donor_data - mean_list)/np.sqrt(var_list)
        self.target_data = (self.target_data - mean_list)/np.sqrt(var_list)
        # self.donor_data = self.donor_data/np.sqrt(var_list)
        # self.target_data = self.target_data/np.sqrt(var_list)

    def apply_weights(self):
        if (self.weighting == "normalize"):
            mean_list = self.donor_data.mean(axis=0)
            var_list = self.donor_data.var(axis=0)
            self.normalize_col(mean_list, var_list)

        elif (self.weighting == "normalize_batch"):
            mean_list = []
            var_list = []
            for k in range(self.num_k):
                donor_batch = self.donor_data.iloc[:,k*self.total_index:(k+1)*self.total_index]
                mean = np.mean(donor_batch.values.flatten())
                var = np.var(donor_batch.values.flatten())
                mean_list = mean_list + [mean] * self.total_index
                var_list = var_list + [var] * self.total_index
            mean_list = pd.Series(mean_list)
            var_list = pd.Series(var_list)
            self.normalize_col(mean_list, var_list)
    
    # de-normilize the target and donor
    def remove_wiehgts(self):
        mean_list = self.weights[0]
        var_list = self.weights[1]
        self.donor_data = (self.donor_data * np.sqrt(var_list)) +  mean_list
        self.target_data = (self.target_data * np.sqrt(var_list)) + mean_list

    def _assignData(self, metrics, pred_interval=1, weighting="normalize", mat_form_method = "fixed", skipNan = True):
        self.metrics = metrics
        self.num_k = len(self.metrics)
        self.weighting = weighting

        self.pred_interval = pred_interval
        self.total_index = self.interv_index + self.pred_interval
        
        self.mat_form_method = mat_form_method
        self.skipNan = skipNan
        
        self.target_data, self.nan_index = self.target.concat(self.metrics)

        self.donor_data = self.donor.concat(self.metrics, self.total_index, self.mat_form_method, self.skipNan, self.nan_index)
        self.donor_data = self.donor_data.iloc[self.donor_data.index != self.target.key] # remove target from the donor (sometimes it's not necessary)

        if (self.donor_data.shape[0] < 5):
            raise Exception("Donor pool size too small. Donor pool size: "+ self.target.key +str(self.donor_data.shape))

        if (self.weighting != None):
            self.apply_weights()

    def fit_threshold(self, metrics, pred_interval=1, threshold =0.99, donorSetup= [None,"fixed", True] , denoiseSetup = ["SVD", "all"], regression_method = "pinv", verbose = False):
        weighting = donorSetup[0] # None / "normalize"
        mat_form_method = donorSetup[1] # "fixed"
        skipNan = donorSetup[2] # (Boolean)
        
        denoise_method = denoiseSetup[0] # "SVD"
        denoise_mat_method = denoiseSetup[1] # "all"

        """
        singvals = (int) the number of singular values to keep; 0 if no HSVT
        mat_form_method = (string) 'fixed' or 'sliding'
        denoise_method = (string) 'svd' or 'als'
        denoise_mat_method = (string) 'all' or 'pre'
        regression_method = (string) 'pinv' or 'lr' or 'lasso'
        skipNan = (boolean) True if we skip the nan in the data (dormant year does not count in)
                            False if we keep the target's nan and construct donor matrix,
                                  and then remove the NaN columns after.
        """
        self._assignData(metrics, pred_interval, weighting, mat_form_method, skipNan)
        
        # compute approximate rank
        if (denoise_mat_method == "all"):
            singvals = utils.approximate_rank(self.donor_data, threshold)
        elif (denoise_mat_method == "pre"):
            donor_pre = utils.get_preint_data(self.donor_data, self.interv_index, self.total_index, self.num_k, reindex = True)
            # print(self.donor_data.shape)
            # print(donor_pre.shape)
            singvals = utils.approximate_rank(donor_pre, threshold)
        
        # denoise & learn weights
        if (denoise_method == "SVD"):
            self.model = SVDmodel(self.num_k, singvals, self.target_data, self.donor_data, self.interv_index, self.total_index, denoise_mat_method, regression_method, skipNan, self.p)
            self.model.fit(verbose)

        elif (denoise_method == "ALS"):
            print("not ready yet")
#             self.model = ALSModel(self.kSingularValues, self.N, self.M, probObservation=self.p, otherSeriesKeysArray=self.otherSeriesKeysArray, includePastDataOnly=False)
        else:
            raise ValueError("Invalid denoise method. Should be 'SVD' or 'ALS'.")
            
        # de-normilize the target and donor
        if (self.weighting != None):
            self.remove_wiehgts()

#     def fit(self, metrics, weights, pred_interval=1, singvals =999, setup = ["fixed", True, "SVD", "all", "pinv"]):
        
#         if (len(weights) != len(metrics)):
#             raise Exception("The length of weights should match with the length of metrics (=num_k).")
#         mat_form_method = setup[0] # "fixed"
#         skipNan = setup[1]
#         denoise_method = setup[2] # "SVD"
#         denoise_mat_method = setup[3] # "all"
#         regression_method = setup[4] #'pinv'

#         """
#         singvals = (int) the number of singular values to keep; 0 if no HSVT
#         mat_form_method = (string) 'fixed' or 'sliding'
#         denoise_method = (string) 'svd' or 'als'
#         denoise_mat_method = (string) 'all' or 'pre'
#         regression_method = (string) 'pinv' or 'lr' or 'lasso'
#         skipNan = (boolean) True if we skip the nan in the data (dormant year does not count in)
#                             False if we keep the target's nan and construct donor matrix,
#                                   and then remove the NaN columns after.
#         """
#         self._assignData(metrics, weights, pred_interval, mat_form_method, skipNan)
        
#         # denoise & learn weights
#         if (denoise_method == "SVD"):
#             self.model = SVDmodel(weights, singvals, self.target_data, self.donor_data, self.interv_index, self.total_index, setup, self.p)
#             self.model.fit()

#         elif (denoise_method == "ALS"):
#             print("not ready yet")
# #             self.model = ALSModel(self.kSingularValues, self.N, self.M, probObservation=self.p, otherSeriesKeysArray=self.otherSeriesKeysArray, includePastDataOnly=False)
#         else:
#             raise ValueError("Invalid denoise method. Should be 'SVD' or 'ALS'.")

    def predict(self):
        """
        donor_post = (df) donor data after the intervention point
        df_return = (df) rows: metrics, cols: calendar year. Contains predicted values.
        """
        donor_post = utils.get_postint_data(combinedDF = self.model.donor_data, intervIndex = self.interv_index, totalIndex = self.total_index, nbrMetrics = self.num_k, reindex = True) 
        pred = np.dot(donor_post.T, self.model.beta)

        # post-treatment to go back to the original status
        if (self.weighting != None):
            mean_post = utils.get_postint_data(combinedDF = self.weights[0].to_frame().T, intervIndex = self.interv_index, totalIndex = self.total_index, nbrMetrics = self.num_k, reindex = True)
            var_post = utils.get_postint_data(combinedDF = self.weights[1].to_frame().T, intervIndex = self.interv_index, totalIndex = self.total_index, nbrMetrics = self.num_k, reindex = True)
            # print("mean: ")
            # print(mean_post)
            # print("var:  ")
            # print(var_post)
            pred = (pred * np.sqrt(var_post.T.values))+ mean_post.T.values

        pred = pred.flatten()
        df_return = pd.DataFrame(index = self.metrics, columns = range(self.pred_interval))
        for k in range(self.num_k):
            df_return.iloc[k,:] = pred[k*self.pred_interval: (k+1)*self.pred_interval]
        return df_return

    def getTrue(self):
        """
        donor_post = (df) donor data after the intervention point
        df_return = (df) rows: metrics, cols: calendar year. Contains true values.
        """
        target_post = utils.get_postint_data(combinedDF = self.target_data, intervIndex = self.interv_index, totalIndex = self.total_index, nbrMetrics = self.num_k, reindex = True)
        df_return = pd.DataFrame(index = self.metrics, columns = range(self.pred_interval))
        for k in range(self.num_k):
            df_return.iloc[k,:] = target_post.iloc[:,k*self.pred_interval: (k+1)*self.pred_interval].values
        return df_return

    def plot(self):
        for j in range(len(self.metrics)):
            metric = self.metrics[j]
            true_trajectory = self.target.data[metric].dropna(axis='columns').iloc[:,:self.total_index]
            print(self.donor_data.iloc[:,j*self.model.total_index:((j+1)*self.total_index)].T)
            pred_val = np.dot(self.donor_data.iloc[:,j*self.model.total_index:((j+1)*self.total_index)].T, self.model.beta).T
            pred_trajectory = pd.DataFrame(pred_val, columns =true_trajectory.columns, index=true_trajectory.index)

            markers_on = [true_trajectory.shape[1]-self.pred_interval]
            plt.plot(true_trajectory.T, marker = 'o', color='red', label = "true")
            plt.plot(pred_trajectory.T, marker = 'o', markevery=markers_on, color='blue', label="prediction")
            plt.xlabel("years played in NBA")
            plt.ylabel(metric)
            plt.title(self.target.key)
            plt.legend()
            plt.show()        
