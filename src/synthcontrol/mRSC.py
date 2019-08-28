################################################################
#
# MultiDimensional Robust Synthetic Control (mRSC)
#
################################################################
import numpy as np
import pandas as pd

from mrsc.src.model.SVDmodel import SVDmodel
from mrsc.src.model.Target import Target
from mrsc.src.model.Donor import Donor

import mrsc.src.utils as utils

class mRSC:
    def __init__(self, donor, target, probObservation=1): 
        """
        donor = (Donor)
        target = (Target)
        probObservation = (float) a value between 0 and 1
        """
        self.donor = donor
        self.target = target
        self.p = probObservation
        
        """
        metrics = (list) list of metrics in donor/target matrix
        num_k = (int) number of metrics
        pred_year = (int) hypothetical year that we live in
        pred_length = (int) how many time points do we want to predict?
        
        mat_form_method = (string) 'fixed' or 'sliding'
        
        target_data = (df) target_matrix
        donor_data = (df) donor matrix
        
        total_index = (int) total size of the full timeline (T)
        interv_index = (int) intervention index to split trian/test (T_0)
        model = SVD or ALS model
        """
        
        self.metrics = None
        self.num_k = None
        self.weights = None
        self.pred_year = None
        self.pred_length = None
        
        self.mat_form_method = None
        
        self.target_data = None
        self.donor_data = None
        self.total_index = None
        self.interv_index = None
        self.model = None
        # self.model.beta has the weights learned
        
    def _assignData(self, metrics, weights, pred_year, pred_length=1, mat_form_method = "fixed"):
        self.metrics = metrics
        self.num_k = len(self.metrics)
        self.weights = weights
        self.pred_year = pred_year
        self.pred_length = pred_length
        
        self.mat_form_method = mat_form_method
        
        self.target_data = self.target.concat(self.metrics, self.pred_year, self.pred_length)
        self.total_index = int(self.target_data.shape[1] / self.num_k)
        self.interv_index = self.total_index - self.pred_length
        
        self.donor_data = self.donor.concat(self.metrics, self.pred_year, self.total_index, self.mat_form_method)
        self.donor_data = self.donor_data.iloc[self.donor_data.index != self.target.key] # remove target from the donor (maybe it's not necessary)

        if (self.donor_data.shape[0] < 2):
            raise Exception("Donor pool size too small. Donor pool size: "+ self.target.key +str(self.donor_data.shape))
        
    def fit(self, metrics, weights, pred_year, pred_length=1, singvals =999, setup = ["fixed", "SVD", "all", "pinv", False]):
        
        if (len(weights) != len(metrics)):
            raise Exception("The length of weights should match with the length of metrics (=num_k).")

        mat_form_method = setup[0] # "fixed"
        denoise_method = setup[1] # "SVD"
        denoise_mat_method = setup[2] # "all"
        regression_method = setup[3] #'pinv'
        skipNan = setup[4]

        """
        singvals = (int) the number of singular values to keep; 0 if no HSVT
        mat_form_method = (string) 'fixed' or 'sliding'
        denoise_method = (string) 'svd' or 'als'
        denoise_mat_method = (string) 'all' or 'pre'
        regression_method = (string) 'pinv' or 'lr' or 'lasso'
        skipNan = (boolean) False if we skip the nan in the data, 
                            True if we remove the target's nan and shift everything left.
        """
        self._assignData(metrics, weights, pred_year, pred_length, mat_form_method)
        
        # denoise & learn weights
        if (denoise_method == "SVD"):
            self.model = SVDmodel(weights, singvals, self.target_data, self.donor_data, self.interv_index, self.total_index, setup, self.p)
            self.model.fit()

        elif (denoise_method == "ALS"):
            print("not ready yet")
#             self.model = ALSModel(self.kSingularValues, self.N, self.M, probObservation=self.p, otherSeriesKeysArray=self.otherSeriesKeysArray, includePastDataOnly=False)
        else:
            raise ValueError("Invalid denoise method. Should be 'SVD' or 'ALS'.")

    def fit_threshold(self, metrics, weights, pred_year, pred_length=1, threshold =0.99, setup = ["fixed", "SVD", "all", "pinv", False]):
        
        if (len(weights) != len(metrics)):
            raise Exception("The length of weights should match with the length of metrics (=num_k).")

        mat_form_method = setup[0] # "fixed"
        denoise_method = setup[1] # "SVD"
        denoise_mat_method = setup[2] # "all"
        regression_method = setup[3] #'pinv'
        skipNan = setup[4]

        """
        singvals = (int) the number of singular values to keep; 0 if no HSVT
        mat_form_method = (string) 'fixed' or 'sliding'
        denoise_method = (string) 'svd' or 'als'
        denoise_mat_method = (string) 'all' or 'pre'
        regression_method = (string) 'pinv' or 'lr' or 'lasso'
        skipNan = (boolean) False if we skip the nan in the data, 
                            True if we remove the target's nan and shift everything left.
        """
        self._assignData(metrics, weights, pred_year, pred_length, mat_form_method)

        # compute approximate rank
        if (denoise_mat_method == "all"):
            singvals = utils.approximate_rank(self.donor_data, threshold)
        elif (denoise_mat_method == "pre"):
            donor_pre = utils.get_preint_data(self.donor_data, self.interv_index, self.total_index, self.num_k, reindex = True)
            # print(self.donor_data.shape)
            # print(donor_pre.shape)
            singvals = utils.approximate_rank(donor_pre, threshold)
        
        # u, s, v = np.linalg.svd(self.donor_data, full_matrices=False)
        # for k in range(1,10,1):
        #     print(k, " : ", np.sum(s[:k]**2) / np.sum(s ** 2))

        # print("threshold: ", threshold)
        # print("singvals : ",singvals)
        
        # denoise & learn weights
        if (denoise_method == "SVD"):
            self.model = SVDmodel(weights, singvals, self.target_data, self.donor_data, self.interv_index, self.total_index, setup, self.p)
            self.model.fit()

        elif (denoise_method == "ALS"):
            print("not ready yet")
#             self.model = ALSModel(self.kSingularValues, self.N, self.M, probObservation=self.p, otherSeriesKeysArray=self.otherSeriesKeysArray, includePastDataOnly=False)
        else:
            raise ValueError("Invalid denoise method. Should be 'SVD' or 'ALS'.")
        
    def predict(self):
        """
        donor_post = (df) donor data after the intervention point
        df_return = (df) rows: metrics, cols: calendar year. Contains predicted values.
        """
        donor_post = utils.get_postint_data(combinedDF = self.donor_data, intervIndex = self.interv_index, totalIndex = self.total_index, nbrMetrics = self.num_k, reindex = True) 
        df_pred = np.dot(donor_post.T, self.model.beta)

        df_return = pd.DataFrame(index = self.metrics, columns = range(self.pred_year, self.pred_year + self.pred_length, 1))
        for k in range(self.num_k):
            df_return.iloc[k,:] = df_pred[k*self.pred_length: (k+1)*self.pred_length]
        
        return df_return

    def getTrue(self):
        """
        donor_post = (df) donor data after the intervention point
        df_return = (df) rows: metrics, cols: calendar year. Contains true values.
        """
        target_post = utils.get_postint_data(combinedDF = self.target_data, intervIndex = self.interv_index, totalIndex = self.total_index, nbrMetrics = self.num_k, reindex = True)
        df_return = pd.DataFrame(index = self.metrics, columns = range(self.pred_year, self.pred_year + self.pred_length, 1))
        for k in range(self.num_k):
            df_return.iloc[k,:] = target_post.iloc[:,k*self.pred_length: (k+1)*self.pred_length].values
        return df_return        
