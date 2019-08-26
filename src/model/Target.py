######################################################
#
# Target
#
######################################################
import numpy as np
import pandas as pd

class Target:
    def __init__(self, key, data, years):
        """
        key: (string) target key
        data: (dict) dictionary of {metric : df} - df has only one row, the index should match with self.key
        years: (df) cols = year_count, index = self.key, values = calendar year
        """
        self.key = key
        self.data = getPlayerFromDict([key], data)
        self.years = years[years.index == key]
        
        self.start_calendar_year = self.years.iloc[0,0]
        self.max_year_count = self.years.dropna(axis=1).columns[-1]
    
    def concat(self, metrics, pred_year):
        """
        metrics: (list) metrics you want to use
        pred_year: (int) the year you want to make a prediction
        
        output: (df) concatenated df for your metrics of choice, up until (and including) the prediction year
        """
        pred_year_count = pred_year - self.start_calendar_year + 1
        df_concat = pd.DataFrame()
        for metric in metrics:
            df_concat = pd.concat([df_concat,self.data[metric].iloc[:,:pred_year_count.astype(int)]],axis=1)
        return df_concat
