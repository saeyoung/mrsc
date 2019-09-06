######################################################
#
# Target
#
######################################################
import numpy as np
import pandas as pd

import mrsc.src.utils as utils

class Target:
    def __init__(self, key, data, years):
        """
        key: (string) target key
        data: (dict) dictionary of {metric : df} - df has only one row, the index should match with self.key
        years: (df) cols = year_count, index = self.key, values = calendar year
        """
        self.key = key
        self.data = utils.getIndivFromDict([key], data)
        self.years = years[years.index == key]
        
        self.start_calendar_year = self.years.iloc[0,0]
        self.max_year_count = self.years.dropna(axis=1).columns[-1]

    def concat(self, metrics, pred_year, pred_length=1):
        """
        metrics: (list) metrics you want to use
        pred_year: (int) the year you want to make a prediction
        
        output: (df) concatenated df for your metrics of choice, up until (and including) the prediction year
        """
        max_index = pred_year - self.start_calendar_year + pred_length
        df_concat = pd.DataFrame()
        for metric in metrics:
            df_concat = pd.concat([df_concat,self.data[metric].iloc[:,:max_index.astype(int)]],axis=1)
        df_concat.columns = range(df_concat.shape[1])
        return df_concat

    def dict(self, metrics, pred_year, pred_length=1):
        """
        metrics: (list) metrics you want to use
        pred_year: (int) the year you want to make a prediction
        
        output: (df) concatenated df for your metrics of choice, up until (and including) the prediction year
        """
        max_index = pred_year - self.start_calendar_year + pred_length
        dict_return = {}
        for metric in metrics:
            dict_return.update({metric: self.data[metric].iloc[:,:max_index.astype(int)]})
        return dict_return
