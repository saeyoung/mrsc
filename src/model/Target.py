######################################################
#
# Target
#
######################################################
import numpy as np
import pandas as pd

import mrsc.src.utils as utils

class Target:
    def __init__(self, key, data):
        """
        key: (string) target key
        data: (dict) dictionary of {metric : df} - df has only one row, the index should match with self.key
        years: (df) cols = year_count, index = self.key, values = calendar year
        """
        self.key = key
        self.data = utils.getIndivFromDict([key], data) # (dict)
        self.total_index = list(self.data.values())[0].dropna(axis=1).shape[1]

    # def addFutureData(pred_interval, values = None):
    #     self.total_index = self.total_index + pred_interval
        
    #     # add values for future timepoints to self.data
    #     if values == None:
    #         values = [0] * pred_interval


    def concat(self, metrics):
        """
        metrics: (list) metrics you want to use
        
        output: (df) concatenated df for your metrics of choice, up until (and including) the prediction year
        """
        df_concat = pd.DataFrame()
        for metric in metrics:
            df_concat = pd.concat([df_concat,self.data[metric].dropna(axis=1)],axis=1)
            
        nan_index = []
        max_index = list(self.data.values())[0].T.last_valid_index() + 1
        df_metric = self.data[metrics[0]].iloc[:,:max_index]
        nan_index = list(df_metric.columns[df_metric.isnull().values.flatten()])

        # if (skipNan == True):
        #     # this skips the NaN values
        #     for metric in metrics:
        #         df_concat = pd.concat([df_concat,self.data[metric].dropna(axis=1)],axis=1)
                
        # else:
        #     # this keeps the NaN values
        #     max_index = list(self.data.values())[0].T.last_valid_index() + 1
        #     for metric in metrics:
        #         df_metric = self.data[metric].iloc[:,:max_index]
        #         df_concat = pd.concat([df_concat,df_metric],axis=1)
        #         nan_index = list(df_metric.columns[df_metric.isnull().values.flatten()])

        df_concat.columns = range(df_concat.shape[1]) # reindex
        return df_concat, nan_index

    # def dict(self, metrics, pred_year, pred_length=1):
    #     """
    #     metrics: (list) metrics you want to use
    #     pred_year: (int) the year you want to make a prediction
        
    #     output: (dict) dict of df for your metrics of choice, up until (and including) the prediction year, {metric: df}
    #     """
    #     max_index = pred_year - self.start_calendar_year + pred_length
    #     dict_return = {}
    #     for metric in metrics:
    #         dict_return.update({metric: self.data[metric].iloc[:,:max_index.astype(int)]})
    #     return dict_return
