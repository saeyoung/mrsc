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

# from tslib.saeyoung.utils import *

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
        
        window = (int) window size
        interv_index = (int) intervention index to split trian/test
        model = SVD or ALS model
        """
        
        self.metrics = None
        self.num_k = None
        self.pred_year = None
        self.pred_length = None
        
        self.mat_form_method = None
        
        self.target_data = None
        self.donor_data = None
        self.window = None
        self.interv_index = None
        self.model = None
        # delf.model.beta has the weights learned
        
    def _assignData(self, metrics, pred_year, pred_length=1, mat_form_method = "fixed"):
        self.metrics = metrics
        self.num_k = len(self.metrics)
        self.pred_year = pred_year
        self.pred_length = pred_length
        
        self.mat_form_method = mat_form_method
        
        self.target_data = self.target.concat(self.metrics, self.pred_year, self.pred_length)
        self.window = int(self.target_data.shape[1] / self.num_k)
        self.interv_index = self.window - self.pred_length
        
        self.donor_data = self.donor.concat(self.metrics, self.pred_year, self.window, self.mat_form_method)
        self.donor_data = self.donor_data.iloc[self.donor_data.index != self.target.key] # remove target from the donor
        
    def fit(self, metrics, pred_year, pred_length=1, singvals =999,  mat_form_method = "fixed", denoise_method = "SVD", denoise_mat_method = "all",regression_method = 'pinv'):
        """
        singvals = (int) the number of singular values to keep; 0 if no HSVT
        mat_form_method = (string) 'fixed' or 'sliding'
        denoise_method = (string) 'svd' or 'als'
        denoise_mat_method = (string) 'all' or 'pre'
        regression_method = (string) 'pinv' or 'lr' or 'lasso'
        """
        self._assignData(metrics, pred_year, pred_length, mat_form_method = "fixed")
        
        # denoise & train test split
        if (denoise_method == "SVD"):
            self.model = SVDmodel(singvals, self.target_data, self.donor_data, self.num_k, self.interv_index, self.window, denoise_mat_method, regression_method, self.p)
            self.model.fit()

        elif (denoise_method == "ALS"):
            print("not ready yet")
#             self.model = ALSModel(self.kSingularValues, self.N, self.M, probObservation=self.p, otherSeriesKeysArray=self.otherSeriesKeysArray, includePastDataOnly=False)
        else:
            raise ValueError("Invalid denoise method. Should be 'SVD' or 'ALS'.")
    
    def get_postint_data(self, combinedDF, intervIndex, totalIndex, nbrMetrics, reindex = True):
        
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
            indexToChoose = indexToChoose + list(range(k*totalIndex + intervIndex, (k+1)*totalIndex))
        return combinedDF.loc[:,indexToChoose]
        
    def predict(self, donor_post):
        """
        donor_post = (df) donor data after the intervention point
        """
        df_pred = np.dot(donor_post.T, self.model.beta)
        
        return df_pred
        
        
